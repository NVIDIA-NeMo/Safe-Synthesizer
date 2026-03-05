# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Statistical functions for synthetic data evaluation.

Provides distribution comparison (Jensen-Shannon), correlation matrix
computation (Pearson, Theil's U, Correlation Ratio), PCA, and
data-overlap detection used by the evaluation components.
"""

import math
import uuid
import warnings

import numpy as np
import pandas as pd
from category_encoders.count import CountEncoder
from joblib import Parallel, delayed
from pandas.api.types import is_numeric_dtype
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", module="dython")
warnings.filterwarnings("ignore", module="dython.nominal")
from dython.nominal import correlation_ratio, theils_u  # noqa

_DEFAULT_JOB_COUNT = 4
_DEFAULT_REPLACE_VALUE = 0.0
UNIQUENESS_THRESHOLD = 0.1


def count_memorized_lines(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    """Count exact row matches between training and synthetic data.

    Args:
        df1: Training dataframe.
        df2: Synthetic dataframe.

    Returns:
        Number of rows present in both dataframes after deduplication.
    """

    # Look for cases where col is numeric in one df, object in the other. Attempt to cast to float.
    def _uptype_object_to_float(
        l: pd.DataFrame,  # noqa: E741
        r: pd.DataFrame,  # noqa: E741
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        for col in set(l.columns).intersection(set(r.columns)):
            if is_numeric_dtype(l[col]) or is_numeric_dtype(r[col]):
                if not is_numeric_dtype(l[col]) or not is_numeric_dtype(r[col]):
                    try:
                        l = l.astype({col: "float"})  # noqa: E741
                        r = r.astype({col: "float"})  # noqa: E741
                    except Exception:
                        # In particular ValueErrors if the non-numeric is not convertible, but catch everything.
                        pass
        return l, r

    # Convert any numeric fields within a df to 'float' first.
    def _floatify(df: pd.DataFrame) -> pd.DataFrame:
        conversions = {}
        for col in df.columns:
            if is_numeric_dtype(df[col]):
                conversions[col] = "float"
        return df.astype(conversions)

    # If one col in a df is numeric and the corresponding one in the other df is NOT, cast to 'object'.
    def _objectify(
        l: pd.DataFrame,  # noqa: E741
        r: pd.DataFrame,  # noqa: E741
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        conversions = {}
        for col in l.columns:
            if col in r.columns:
                if not is_numeric_dtype(l[col]) or not is_numeric_dtype(r[col]):
                    conversions[col] = "object"
        return l.astype(conversions), r.astype(conversions)

    # Do the casts.
    l, r = _uptype_object_to_float(df1, df2)  # noqa: E741
    l, r = _objectify(_floatify(l), _floatify(r))  # noqa: E741

    # Do an inner join on the intersection of columns present in both dfs.
    inner_join = pd.merge(l.drop_duplicates(), r.drop_duplicates())

    return len(inner_join)


def get_categorical_field_distribution(field: pd.Series) -> dict:
    """Compute the normalized value-count distribution of a categorical column.

    Args:
        field: Column series to analyze.

    Returns:
        Mapping of ``{value_str: percentage}`` where percentages are in ``[0, 100]``.
    """
    distribution = {}
    if len(field) > 0:
        for v in field:
            distribution[str(v)] = distribution.get(str(v), 0) + 1
        series_len = float(len(field))
        for k in distribution.keys():
            distribution[k] = distribution[k] * 100 / series_len
    return distribution


def get_numeric_distribution_bins(training: pd.Series, synthetic: pd.Series):
    """Compute shared histogram bin edges for two numeric series.

    Uses the ``"doane"`` strategy on the combined data, falling back to
    500 fixed bins if the result is empty or too large.

    Args:
        training: Numeric series from the training dataframe.
        synthetic: Numeric series from the synthetic dataframe.

    Returns:
        Array of bin edges spanning both series.
    """
    training = training.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    # Numeric data. Want the same bins between both df's. We bin based on scrubbed data.
    if len(training) == 0:
        min_value = np.nanmin(synthetic)
        max_value = np.nanmax(synthetic)
    elif len(synthetic) == 0:
        min_value = np.nanmin(training)
        max_value = np.nanmax(training)
    else:
        min_value = min(np.nanmin(training), np.nanmin(synthetic))
        max_value = max(np.nanmax(training), np.nanmax(synthetic))
    bins = np.array([], dtype=np.float64)

    # Use 'doane' to find bins. 'fd' causes too many OOM issues.
    # We also bin across the training and synthetic Series combined since we are binning across the combined range, otherwise we can see OOM's or sigkill's.
    try:
        bins = np.histogram_bin_edges(pd.concat([training, synthetic]), bins="doane", range=(min_value, max_value))
    except Exception:
        pass
    # If 'doane' still doesn't do the trick just force 500 bins.
    if len(bins) == 0 or len(bins) > 500:
        try:
            bins = np.histogram_bin_edges(pd.concat([training, synthetic]), bins=500, range=(min_value, max_value))
        except Exception:
            pass
    return bins


def get_numeric_field_distribution(field: pd.Series, bins) -> dict:
    """Compute the normalized distribution of a numeric column cut into bins.

    Args:
        field: Numeric column series.
        bins: Bin edges (typically from ``get_numeric_distribution_bins``).

    Returns:
        Mapping of ``{bin_label: proportion}`` where proportions are in ``[0, 1]``.
    """
    binned_data = pd.cut(field, bins, include_lowest=True)
    distribution = {}
    for d in binned_data:
        if str(d) != "nan":
            distribution[str(d)] = distribution.get(str(d), 0) + 1
    field_length = len(binned_data)
    for k in distribution.keys():
        distribution[k] = distribution[k] / field_length
    return distribution


def compute_distribution_distance(d1: dict, d2: dict) -> float:
    """Compute the Jensen-Shannon distance between two distributions.

    Args:
        d1: First distribution dict (values are a probability vector).
        d2: Second distribution dict.

    Returns:
        JS distance in ``[0, 1]``. Returns ``0.5887`` if either distribution
        sums to zero.
    """
    all_keys = set(d1.keys()).union(set(d2.keys()))
    if len(all_keys) == 0:
        return 0.0
    d1_values = []
    d2_values = []
    for k in all_keys:
        d1_values.append(d1.get(k, 0.0))
        d2_values.append(d2.get(k, 0.0))
    sd1 = sum(d1_values)
    sd2 = sum(d2_values)
    if sd1 == 0 or np.isnan(sd1) or sd2 == 0 or np.isnan(sd2):
        return 0.5887
    return float(jensenshannon(np.asarray(d1_values), np.asarray(d2_values), base=2))


def calculate_pearsons_r(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, opt: bool) -> tuple[float, float]:
    """Compute the Pearson correlation coefficient for a column pair.

    Args:
        x: First input array.
        y: Second input array.
        opt: If ``False``, drop rows where either value is NaN before
            computing.  If ``True``, assume NaNs have already been replaced.

    Returns:
        Tuple of (Pearson r, two-tailed p-value).
    """
    if not opt:
        # drop missing values, when either the x or y value is null/nan
        arr = (
            pd.DataFrame(np.array([x, y]).transpose(), columns=["x", "y"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis="index", how="any")
        )
        conditions = [len(arr["x"]) <= 1, len(arr["y"]) <= 1, arr["x"].nunique() <= 1, arr["y"].nunique() <= 1]
        if any(conditions):
            return 0.0, 0.0
        return pearsonr(arr["x"], arr["y"])
    else:
        # else we've already replaced nan's with 0's for entire datafile
        return pearsonr(x, y)


def calculate_correlation_ratio(x: pd.Series, y: pd.Series, opt: bool) -> float:
    """Compute the Correlation Ratio for a categorical-numeric column pair.

    Args:
        x: Categorical input array.
        y: Numeric input array.
        opt: If ``False``, drop rows where ``y`` is NaN before computing.

    Returns:
        Correlation ratio in ``[0, 1]``.
    """
    if not opt:
        # Drop missing values if y (the numeric column) is null/nan
        df = pd.DataFrame({"x": x, "y": y}).replace(to_replace=[np.inf, -np.inf], value=np.nan, inplace=False).dropna()
        x = df["x"]
        y = df["y"]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    else:
        # Either way, we've dealt with missing values by now, so tell dython not to do anything
        return correlation_ratio(x, y, nan_strategy="none")


def calculate_theils_u(x, y):
    """Compute Theil's U (uncertainty coefficient) for two categorical columns.

    Args:
        x: First categorical array.
        y: Second categorical array.

    Returns:
        Theil's U in ``[0, 1]``.
    """
    # Drop missing values if x or y is null/nan
    df = pd.DataFrame({"x": x, "y": y})
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    x = df["x"]
    y = df["y"]
    if len(x) == 0 or len(y) == 0:
        return 0
    else:
        return theils_u(x, y, nan_strategy="none")


def calculate_correlation(
    df: pd.DataFrame,
    nominal_columns: list[str] | None = None,
    job_count: int = _DEFAULT_JOB_COUNT,
    opt: bool = False,
) -> pd.DataFrame:
    """Build a full correlation matrix using Pearson, Theil's U, and Correlation Ratio.

    Numeric-numeric pairs use Pearson's r, categorical-categorical pairs use
    Theil's U, and categorical-numeric pairs use Correlation Ratio (or
    Theil's U for highly-unique categoricals).

    Args:
        df: Input dataframe.
        nominal_columns: Columns to treat as categorical.
        job_count: Number of parallel jobs for pairwise computations.
        opt: If ``True``, globally replace NaNs with ``0`` for speed
            (slightly less accurate).

    Returns:
        Square correlation dataframe indexed and columned by ``df.columns``.
    """
    # PLAT-1131 Ensure that all nominal columns are present in df.
    if nominal_columns is not None:
        _df_cols = df.columns
        nominal_columns = [col for col in nominal_columns if col in _df_cols]
    # If opt is True, then go the faster (just not quite as accurate) route of global replace missing with 0
    if opt:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(_DEFAULT_REPLACE_VALUE, inplace=True)

    columns = df.columns
    if nominal_columns is None:
        nominal_columns = list()

    df_cp = df.copy()
    # Convert pd.Int64Dtype() and boolean columns to object dtype to handle NaN values with NaN proxies.
    pandas_dtypes = df_cp[nominal_columns].columns[
        df_cp[nominal_columns].dtypes.apply(lambda x: x.name in [pd.BooleanDtype(), pd.Int64Dtype()])
    ]

    df_cp[pandas_dtypes] = df_cp[pandas_dtypes].astype("object")
    # Replace NaNs with NaN proxy only for nominal columns. This helps with more consistency in FCS regardless of the generated row counts.
    df_cp[nominal_columns] = df_cp[nominal_columns].fillna(f"safe-synthesizer-{uuid.uuid4().hex}")

    corr = np.zeros((len(columns), len(columns)))
    single_value_columns = []
    numeric_columns = []

    column_to_index = {}

    # Set up all the column groupings needed for correlation
    for i, c in enumerate(columns):
        if df_cp[c].nunique() == 1:
            single_value_columns.append(c)
        elif c not in nominal_columns:
            if df_cp[c].dtype == "object":
                nominal_columns.append(c)
            else:
                numeric_columns.append(c)

        column_to_index[c] = i

    # Replace NaNs with NaN proxy one more time since the nominal column might have updated.
    df_cp[nominal_columns] = df_cp[nominal_columns].fillna(f"safe-synthesizer-{uuid.uuid4().hex}")
    nominal = [x for x in nominal_columns if x not in single_value_columns]
    df_rows = df_cp.shape[0]
    high_unique_nominal = []
    completely_unique_nominal = []
    not_high_unique_nominal = []
    uniqueness_ratios = df_cp.nunique() / df_rows
    for c in nominal:
        if uniqueness_ratios[c] == 1:
            completely_unique_nominal.append(c)
        elif uniqueness_ratios[c] > UNIQUENESS_THRESHOLD:
            high_unique_nominal.append(c)
        else:
            not_high_unique_nominal.append(c)

    notcompletely_unique_nominal = [x for x in nominal if x not in completely_unique_nominal]

    # Do Theil's U shortcut for 100% unique nominal (Amy invention that is 99.9% correct, and saves massive time)
    for x in completely_unique_nominal:
        x_index = column_to_index[x]

        corr[x_index, :] = 1.0
        for y in columns:
            y_index = column_to_index[y]
            if x == y:
                corr[y_index][x_index] = 1.0
            # Edge case, guard against ValueError in math.log when the other column is empty
            elif df_cp[y].nunique() == 0:
                corr[y_index][x_index] = 0.0
            else:
                corr[y_index][x_index] = math.log(df_cp[y].nunique()) / math.log(df_cp[x].nunique())

    for x in single_value_columns:
        x_index = column_to_index[x]
        corr[:, x_index] = 0.0
        corr[x_index, :] = 0.0
        corr[x_index, x_index] = 1.0

    # Do nominal-nominal excluding any that are 100% unique (Theil's U)
    scores = Parallel(n_jobs=job_count)(
        delayed(calculate_theils_u)(df_cp[field1], df_cp[field2])
        for field1 in notcompletely_unique_nominal
        for field2 in notcompletely_unique_nominal
    )
    i = 0
    for field1 in notcompletely_unique_nominal:
        field1_index = column_to_index[field1]
        for field2 in notcompletely_unique_nominal:
            field2_index = column_to_index[field2]
            if field1 == field2:
                corr[field1_index][field2_index] = 1.0
            else:
                # looks backward, but is correct
                corr[field2_index][field1_index] = scores[i]
            i += 1

    # Do "not_high_unique_nominal with numeric" (Correlation Ratio)
    scores = Parallel(n_jobs=job_count)(
        delayed(calculate_correlation_ratio)(df_cp[field1], df_cp[field2], opt)
        for field1 in not_high_unique_nominal
        for field2 in numeric_columns
    )
    i = 0
    for field1 in not_high_unique_nominal:
        field1_index = column_to_index[field1]
        for field2 in numeric_columns:
            field2_index = column_to_index[field2]
            corr[field1_index][field2_index] = scores[i]
            corr[field2_index][field1_index] = scores[i]
            i += 1

    # Do high_unique_nominal with numeric (Theil's U) (excluding 100% unique)
    # This fixes the problem of highly unique categorical causing mass instability when using
    # the normal approach of correlation ratio.  Because there are so many categorical buckets
    # many end up with just one number is them, which causes correlation ratio's approach of
    # comparing the mean within buckets to the mean overall to give unstable, over inflated
    # correlation values.  Using Theil's U instead gives a much more realistic score.
    # Because Theil's U is asymmetric, doing x-y and y-x correlation separately.
    # U(nominal|numeric)
    scores_xy = Parallel(n_jobs=job_count)(
        delayed(calculate_theils_u)(df_cp[field1], df_cp[field2])
        for field1 in high_unique_nominal
        for field2 in numeric_columns
    )
    # U(numeric|nominal)
    scores_yx = Parallel(n_jobs=job_count)(
        delayed(calculate_theils_u)(df_cp[field2], df_cp[field1])
        for field1 in high_unique_nominal
        for field2 in numeric_columns
    )
    i = 0
    for field1 in high_unique_nominal:
        field1_index = column_to_index[field1]
        for field2 in numeric_columns:
            field2_index = column_to_index[field2]
            corr[field2_index][field1_index] = scores_xy[i]
            corr[field1_index][field2_index] = scores_yx[i]
            i += 1

    # Do numeric numeric (Pearson's)
    num_len = len(numeric_columns)
    if num_len > 1:
        delayed_calls = []
        for i in range(num_len - 1):
            for j in range(i + 1, num_len):
                delayed_calls.append(
                    delayed(calculate_pearsons_r)(df_cp[numeric_columns[i]], df_cp[numeric_columns[j]], opt)
                )
        scores = Parallel(n_jobs=1)(delayed_calls)
        x = 0
        for i in range(num_len - 1):
            num_columns_index = column_to_index[numeric_columns[i]]
            for j in range(i + 1, num_len):
                num_columns_jindex = column_to_index[numeric_columns[j]]
                corr[num_columns_index][num_columns_jindex] = scores[x][0]
                corr[num_columns_jindex][num_columns_index] = scores[x][0]
                x += 1

    for x in numeric_columns:
        x_index = column_to_index[x]
        corr[x_index][x_index] = 1.0

    corr_final = pd.DataFrame(corr, index=columns, columns=columns)
    corr_final[corr_final == np.inf] = 0
    corr_final.fillna(value=np.nan, inplace=True)

    return corr_final


def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a dataframe for PCA: fill missing values, encode categoricals, and standardize.

    Args:
        df: Raw dataframe to prepare.

    Returns:
        Standardized dataframe (mean 0, std 1) with all columns numeric.
    """
    df_cp = df.copy()
    # Divide the dataframe into numeric and categorical
    nominal_columns = list(df_cp.select_dtypes(include=["object", "category", "boolean"]).columns)
    int64_dtypes = df_cp.columns[df_cp.dtypes == pd.Int64Dtype()]
    # Convert pandas boolean dtype columns to object to replace possible NaNs with "Missing" values:
    boolean_dtypes = df_cp[nominal_columns].columns[df_cp[nominal_columns].dtypes == pd.BooleanDtype()]
    df_cp[boolean_dtypes] = df_cp[boolean_dtypes].astype("object")
    numeric_columns = []
    for c in df_cp.columns:
        if c not in nominal_columns:
            numeric_columns.append(c)
    df_cat = df_cp.reindex(columns=nominal_columns)
    df_num = df_cp.reindex(columns=numeric_columns)
    df_cat_labels = pd.DataFrame()
    # Fill missing values and encode categorical columns by the frequency of each value
    if len(numeric_columns) > 0:
        if len(int64_dtypes) > 0:
            # pd.Int64Dtype only accepts integers, hence convert any float median to an integer.
            df_num[int64_dtypes] = df_num[int64_dtypes].fillna(df_num[int64_dtypes].median().astype(int))

        df_num = df_num.fillna(df_num.median())

    if len(nominal_columns) > 0:
        df_cat = df_cat.fillna("Missing")
        encoder = CountEncoder()
        df_cat_labels = pd.DataFrame(encoder.fit_transform(df_cat))

    # Merge numeric and categorical back into one dataframe
    if len(nominal_columns) == 0:
        new_df = df_num
    elif len(numeric_columns) == 0:
        new_df = df_cat_labels
    else:
        new_df = pd.concat([df_num, df_cat_labels], axis=1, sort=False)

    # Finally, standardize all values
    all_columns = nominal_columns + numeric_columns
    new_df = pd.DataFrame(StandardScaler().fit_transform(new_df), columns=all_columns)

    return new_df


def compute_pca(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """Run PCA on a single dataframe after normalization.

    Args:
        df: Dataframe to decompose.
        n_components: Number of principal components to keep.

    Returns:
        Dataframe with columns ``pc1``, ``pc2``, etc.
    """
    seed = 444
    df_ = df.replace([np.inf, -np.inf], np.nan, inplace=False).dropna(axis="columns", how="all")

    df_norm = normalize_dataset(df_)
    pca = PCA(n_components=n_components, random_state=seed)
    projected = pca.fit_transform(df_norm)
    columns = [f"pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(data=projected, columns=columns)


def compute_joined_pcas(
    reference_df: pd.DataFrame,
    output_df: pd.DataFrame,
    n_components: int = 2,
    include_variance: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run joined PCA: fit on reference, transform both reference and output.

    Args:
        reference_df: Training dataframe (used to fit the scaler and PCA).
        output_df: Synthetic dataframe (transformed only).
        n_components: Number of principal components to keep.
        include_variance: If ``True``, column names include explained variance ratios.

    Returns:
        Tuple of (reference PCA dataframe, output PCA dataframe).
    """
    seed = 444

    # Normalize the train and synthetic dataframes to mean 0 and std 1
    sc = StandardScaler()
    reference_norm = sc.fit_transform(reference_df)
    output_norm = sc.transform(output_df)

    pca = PCA(n_components=n_components, random_state=seed)
    projected_reference = pca.fit_transform(reference_norm)
    projected_output = pca.transform(output_norm)

    if include_variance:
        eigenvalues = pca.explained_variance_ratio_
        columns = [f"pc{i + 1} - variance {eigenvalues[i]:.2f}" for i in range(n_components)]
    else:
        columns = [f"pc{i + 1}" for i in range(n_components)]

    reference_pca = pd.DataFrame(data=projected_reference, columns=columns)
    output_pca = pd.DataFrame(data=projected_output, columns=columns)

    return (reference_pca, output_pca)


def count_missing(df: pd.DataFrame) -> int:
    """Count total missing (NaN/null) values across all cells.

    Args:
        df: Dataframe to inspect.

    Returns:
        Total number of missing values.
    """
    return int(df.isnull().sum().sum())


def percent_missing(df: pd.DataFrame) -> float:
    """Compute the percentage of missing values in a dataframe.

    Args:
        df: Dataframe to inspect.

    Returns:
        Percentage of missing values in ``[0, 100]``.
    """
    r, c = df.shape
    total_cells = r * c
    if total_cells == 0:
        return 0.0
    total_missing = count_missing(df)
    return 100.0 * total_missing / total_cells
