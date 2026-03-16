# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# ruff: noqa

"""Tooltip text displayed in the multi-modal HTML evaluation report."""

tooltips = {
    "dataset_statistics_info": """
        The dataset statistics provide a summary of the datasets. The table includes the number of rows and columns,
        the percentage of missing values in the datasets, and the number of lines from the reference data that are repeated verbatim in the output data.
    """,
    "sqs_info": """
        The SQS (Synthetic Quality Score) is a measure of how well the output data matches the reference data. A higher SQS indicates a better match.
        SQS is comprised of five metrics. Column Correlation Stability, Deep Structure Stability, and Column Distribution Stability are calculated and
        then averaged for any numeric and categorical columns in the datasets. Text Structure Similarity and Text Semantic Similarity are calculated and
        then averaged for any free text columns in the datasets. Finally, to calculate the overall SQS, a weighted average of the numeric and
        categorical metrics and the text metrics is taken based on the number of numeric, categorical, and free text columns in the datasets.
        If there are no columns of the required type, the metrics will show up as N/A.
    """,
    "reference_columns_info": """
        This table shows the count of unique and missing column values, the average length of each column, as well as its data type.
        When a dataset contains a large number of highly unique columns or a large amount of missing data,
        these characteristics can impede the model's ability to accurately learn the statistical structure of the data.
        Exceptionally long columns can also have the same impact.
    """,
    "column_correlation_stability_info": """
        To measure Column Correlation Stability, the correlation between every pair of numeric and categorical columns is computed first in the reference data,
        and then in the output data. The absolute difference between these values is then computed and averaged across all columns.
        The lower this average value is, the higher the Column Correlation Stability quality score will be.
        To aid in the comparison of correlations, a heatmap is shown for both the reference data and the output data,
        as well as a heatmap for the computed difference of correlation values. If the intended purpose of the output data is to
        perform statistical analysis or machine learning, maintaining the integrity of correlations can be critical.
    """,
    "deep_structure_stability_info": """
        Deep Structure Stability is a measure of how well the output data matches the reference data in terms of the structure of the data.
        A higher similarity indicates a better match. The deep structure stability is calculated using the cosine similarity between the
        embeddings of the numeric and categorical columns in the reference and output data.
    """,
    "column_distribution_stability_info": """
        Column Distribution Stability is a measure of how closely the column distributions in the output data mirror those in the reference data.
        For each numeric and categorical column, we use a common approach for comparing two distributions referred to as the Jensen-Shannon (JS) Distance.
        The lower the JS Distance score is on average across all numeric and categorical columns, the higher the column Distribution Stability quality score will be.
        Note, highly unique strings (neither numeric nor categorical) will not have a distributional distance score.
        To aid in the comparison of reference versus output column, a bar chart or histogram is shown for each numeric and categorical column.
        Depending on the intended purpose of the output data, maintaining the integrity of column distributions can be critical.
    """,
    "text_structure_similarity_info": """
        Text Structure Similarity is a measure of how well the output data matches the reference data in terms of the structure of the text for any free text columns that are present.
        A higher similarity indicates a better match. The text structure similarity is calculated using the cosine similarity between the
        word embeddings of the reference and output data for any free text columns.
    """,
    "text_semantic_similarity_info": """
        Text Semantic Similarity is a measure of how well any free text columns in the output data match the reference data in terms of the meaning of the text.
        A higher similarity indicates a better match. The text semantic similarity is calculated using the cosine similarity between the
        sentence embeddings of the reference and output data.
    """,
    "differential_privacy_info": """
    Differential Privacy (DP) is generally regarded as the highest level of privacy, providing mathematical guarantees around the protection of individual training
    records based on the values of the epsilon and delta parameters. Lower epsilon indicates higher levels of privacy. Note that while applying DP increases privacy,
    it may reduce quality and / or increase runtime.
    """,
    "data_privacy_score_info": """
        The Data Privacy Score (DPS) is a measure of how well the output data protects sensitive information present in the reference data.
        A higher score indicates better privacy protection. DPS is comprised of two metrics, Membership Inference Protection and Attribute Inference Protection.
        The scores for each of these metrics are averaged to determine the overall DPS.
    """,
    "pii_replay_info": """
        PII Replay counts the number of total and unique instances of personally identifiable information (PII) from your reference data that show up in your output data.
        Lower counts under "Output Data PII Replay" indicate higher privacy. Note that some PII replay is often expected (and even desirable).<br>
        In general, you can expect entities that are rarer (and therefore have many possible values), like full address or full name,
        to have lower amounts of PII replay than entities that are more common (and therefore have fewer possible values), like first name or US state.
        To reduce the amount of PII replay, perform PII replacement prior to synthesizer.
    """,
    "membership_inference_info": """
        Membership Inference Protection is a measure of how well-protected your data is from membership inference attacks.
        A membership inference attack is a type of privacy attack on machine learning models where an adversary aims
        to determine whether a particular data sample was part of the model's output dataset. By exploiting the differences
        in the model's responses to data points from its training set versus those it has never seen before, an attacker can
        attempt to infer membership. This type of attack can have critical privacy implications, as it can reveal whether
        specific individuals' data was used to train the model. Based on directly analyzing the output data, a high
        score indicates that your output data is well-protected from this type of attack.
    """,
    "attribute_inference_info": """
        Attribute Inference Protection is a measure of how well-protected your data is from attribute inference attacks.
        An attribute inference attack is a type of privacy attack on machine learning models where an adversary seeks
        to infer missing attributes or sensitive information about individuals from their data that was used to train
        the model. By leveraging the model's output, the attacker can attempt to predict unknown attributes of a data sample.
        For a specific attribute, a high score indicates that even when other attributes are known, that specific attribute is difficult to predict.
        This type of attack poses significant privacy risks, as it can uncover sensitive details about individuals that
        were not intended to be revealed by the data owners. Based on directly analyzing the output data, a high
        score indicates that your output data is well-protected from this type of attack.
    """,
}
