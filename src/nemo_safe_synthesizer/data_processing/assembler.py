# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Callable, Generator

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets
from datasets.exceptions import DatasetGenerationError
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from nemo_safe_synthesizer import utils
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.record_utils import (
    extract_records_from_jsonl_string,
    records_to_jsonl,
)
from nemo_safe_synthesizer.data_processing.stats import (
    RunningStatistics,
    Statistics,
)
from nemo_safe_synthesizer.defaults import (
    DEFAULT_CACHE_PREFIX,
    TRAIN_SET_SIZE_BUFFER,
)
from nemo_safe_synthesizer.errors import (
    GenerationError,
    ParameterError,
)
from nemo_safe_synthesizer.holdout.holdout import grouped_train_test_split, naive_train_test_split
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.observability import get_logger

logger = get_logger(__name__)

NUM_SPECIAL_TOKENS = 2
GeneratorType = Generator[dict[str, list], None, None]


def _get_max_tokens_action(rope_scaling_factor: int | None) -> str:
    rsf = rope_scaling_factor if rope_scaling_factor is not None else 1
    if rsf <= 5:
        max_tokens_action = (
            "Training this model will require modifying your dataset and/or the model "
            "configuration. Consider increasing the rope_scaling_factor parameter "
            "(currently set to {rsf}, you could start by increasing "
            "to {rsf_plus_1} (must be an integer value between 1 and 6)), "
            "reducing the number of columns in your dataset, shortening the "
            "column names, filtering out rows with long text values, and/or "
            "reducing the number of rows per sequence if you are using the "
            "group_training_examples_by parameter."
        )
        return max_tokens_action.format(
            rsf=rsf,
            rsf_plus_1=rsf + 1,
        )
    else:
        max_tokens_action = (
            "Training this model will require modifying your dataset. "
            "The rope_scaling_factor is currently set to 6, which cannot be increased further. "
            "Consider reducing the number of columns in your dataset, shortening the "
            "column names, filtering out rows with long text values, and/or "
            "reducing the number of rows per sequence if you are using the "
            "group_training_examples_by parameter."
        )
        return max_tokens_action


class Example:
    """A single training example containing a prompt and records.

    A training example consists of a prompt followed by a "sequence" or
    "sequences" of records, where each sequence is (optionally) enclosed
    by the BOS and EOS special tokens.
    """

    def __init__(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        metadata: ModelMetadata,
    ):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.metadata = metadata

        self.input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if self.metadata.prompt_config.add_bos_token_to_prompt:
            self.input_ids = [self.metadata.prompt_config.bos_token_id] + self.input_ids
        if self.metadata.prompt_config.add_eos_token_to_prompt:
            self.input_ids = self.input_ids + [self.metadata.prompt_config.eos_token_id]

        # We use -100 to ignore the prompt tokens when calculating the loss.
        self.labels = [-100] * len(self.input_ids)
        self.attention_mask = [1] * len(self.input_ids)

        self.num_sequences = 0

    @property
    def num_tokens(self) -> int:
        return len(self.input_ids)

    def add_sequence(self, seq: dict[str, list[int]], add_special_tokens: bool = True) -> None:
        """Add a sequence of records to the example.

        Args:
            seq: Dictionary containing 'input_ids' and 'attention_mask' for the sequence.
            add_special_tokens: Whether to add special tokens to the sequence.

        Raises:
            GenerationError: If the number of tokens in the example exceeds the context length.
        """
        input_ids = (
            [self.metadata.prompt_config.bos_token_id] + seq["input_ids"] + [self.metadata.prompt_config.eos_token_id]
            if add_special_tokens
            else seq["input_ids"]
        )
        attention_mask = [1] + seq["attention_mask"] + [1] if add_special_tokens else seq["attention_mask"]
        self.input_ids.extend(input_ids)
        self.attention_mask.extend(attention_mask)
        self.labels.extend(input_ids)
        self.num_sequences += 1

        if self.num_tokens > self.metadata.max_seq_length:
            max_tokens_action = _get_max_tokens_action(self.metadata.rope_scaling_factor)
            msg = f"The number of tokens in an example exceeds the available context length. {max_tokens_action}"
            logger.error(msg)
            raise GenerationError(msg)

    def to_dict(self) -> dict[str, list]:
        """Converts the example to a dictionary format suitable for training.

        Returns:
            A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "labels": self.labels,
        }


@dataclass
class TrainingExamples:
    """Container for managing a dataset of training examples.

    Attributes:
        train: 🤗 Dataset of the training examples.
        stats: Running statistics calculated during example construction.
        test: 🤗 Dataset of the test examples, if available.
    """

    train: Dataset
    stats: dict[str, Statistics]
    test: Dataset | None = None


class TrainingExampleAssembler(ABC):
    """Base class for assembling LLM training examples.

    Subclasses of this class are responsible for converting a dataset into a
    format suitable for training / fine-tuning LLMs.

    Args:
        dataset: Dataset to be processed.
        tokenizer: Tokenizer used for tokenizing the dataset records.
        metadata: Internal training configuration, e.g., prompt template,
            bos/eos tokens, and where to use them.
        keep_columns: List of columns to keep in the tokenized dataset. This is useful
            if you need certain fields for subsequent processing (e.g., grouping).
        test_size: Absolute number of records you want in the test set. If None
            or 0, there will be no test set and hence no evaluation during training.
        cache_file_path: Path to store the cached dataset for efficient data access.
        seed: Seed for the random number generator and train-test split.
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        metadata: ModelMetadata,
        keep_columns: list[str] | None = None,
        test_size: int | None = None,
        cache_file_path: str | Path | None = None,
        seed: int | None = None,  # TODO: probably should include with metadata!
        *args,
        **kwargs,
    ):
        if test_size is not None and test_size > 0 and test_size > len(dataset) - TRAIN_SET_SIZE_BUFFER:
            msg = (
                "The test set size is too large compared to the input dataset. You must "
                f"have `test_set_size < len(dataset) - {TRAIN_SET_SIZE_BUFFER} records`. "
                f"You gave `test_set_size = {test_size}` and `len(dataset) = {len(dataset)}`. "
                "Please reduce the test set size or provide a larger dataset."
            )
            logger.error(msg)
            raise ParameterError(msg)

        self.metadata = metadata
        self.tokenizer = tokenizer
        self.stats = defaultdict(RunningStatistics)
        self.stats_val = defaultdict(RunningStatistics)
        # adding this extra instead of "" due to hf datasets being weird about the cache path parent dirs -
        # it's erroring out when we pass an empty string, with a 'filenotfound' error.
        fp = Path(cache_file_path) if cache_file_path else Path.cwd()
        self.cache_file_path = fp / f"{DEFAULT_CACHE_PREFIX}_{uuid.uuid4().hex[:5]}"
        self.test_size = test_size
        self.keep_columns = keep_columns or []
        self.seed = seed

        self.schema_prompt = utils.create_schema_prompt(
            dataset.column_names,
            instruction=metadata.instruction,
            prompt_template=metadata.prompt_config.template,
        )

        # The prompt IDs attribute does *not* include special tokens.
        self.schema_prompt_ids: list[int] = tokenizer(self.schema_prompt, add_special_tokens=False)["input_ids"]

        self.tokenized_records = self._tokenize_dataset(dataset, keep_columns)
        processed_dataset = self._preprocess_before_splitting(self.tokenized_records)
        self._apply_train_test_split(processed_dataset)

    @abstractmethod
    def _preprocess_before_splitting(self, tokenized_records: Dataset) -> Dataset:
        """Use case-specific processing before splitting the dataset.

        Example processing is ordering and grouping the records before splitting.
        Standard tabular data without any grouping does not need to perform
        any processing before splitting.
        """

    @abstractmethod
    def assemble_training_examples(self, data_fraction: float = 1.0) -> TrainingExamples:
        """Build examples from the tokenized dataset.

        Args:
            data_fraction: Fraction of the dataset to use for example generation.

        Returns:
            TrainingExamples object containing a 🤗 Dataset objects for the train
            and test set of examples, as well as an object with associated statistics.
        """

    @property
    @abstractmethod
    def num_records_train(self) -> int: ...

    @property
    @abstractmethod
    def num_records_validation(self) -> int: ...

    @property
    def num_records_total(self) -> int:
        return self.num_records_train + self.num_records_validation

    @classmethod
    def from_data(
        cls,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        metadata: ModelMetadata,
        config: SafeSynthesizerParameters,
        test_size: int | None = None,
        seed: int | None = None,
        cache_file_path: str | Path | None = None,
        keep_columns: list[str] | None = None,
        **kwargs,
    ) -> GroupedDataExampleAssembler | TabularDataExampleAssembler:
        if config.data.group_training_examples_by is not None:
            return GroupedDataExampleAssembler(
                group_training_examples_by=config.data.group_training_examples_by,
                order_training_examples_by=config.data.order_training_examples_by,
                dataset=dataset,
                tokenizer=tokenizer,
                metadata=metadata,
                test_size=config.training.validation_ratio,
                seed=seed,
                cache_file_path=cache_file_path,
                keep_columns=keep_columns,
                **kwargs,
            )
        else:
            return TabularDataExampleAssembler(
                dataset=dataset,
                tokenizer=tokenizer,
                metadata=metadata,
                test_size=config.training.validation_ratio,
                seed=seed,
                cache_file_path=cache_file_path,
                keep_columns=keep_columns,
                **kwargs,
            )

    @staticmethod
    def _convert_records_to_jsonl(records: dict[str, list]) -> dict[str, list[str]]:
        """Convert records to JSONL format and return as list of strings in a dict"""
        jsonl = records_to_jsonl(records)
        return {"text": [f"{r}\n" for r in extract_records_from_jsonl_string(jsonl)]}

    def _apply_train_test_split(self, dataset: Dataset) -> None:
        """Split the dataset into training and test sets."""
        if self.test_size is not None and self.test_size > 0:
            split_dataset = naive_train_test_split(
                dataset.to_pandas(), test_size=self.test_size, random_state=self.seed
            )
            train_df, test_df = split_dataset
            self.train_dataset = Dataset.from_pandas(train_df)
            self.validation_dataset = Dataset.from_pandas(test_df)
            self.validation_dataset.info.description += "is_val"

        else:
            self.train_dataset = dataset
            self.validation_dataset = None

    def _order_records(self, dataset: Dataset, order_by: str) -> Dataset:
        """Order the tokenized records in the dataset by the specified column."""
        logger.info(f"Sorting dataset by '{order_by}'")
        return dataset.sort(order_by)

    def _tokenize_records(self, records: dict[str, list]) -> dict[str, list]:
        """Tokenize the records in the dataset and return as a dict of lists."""
        if len(self.schema_prompt_ids) > self.metadata.max_seq_length:
            max_tokens_action = _get_max_tokens_action(self.metadata.rope_scaling_factor)
            msg = (
                "The dataset schema requires more tokens than the max length of the model. "
                "This likely means that the table is too wide to be used with this model. "
                f"{max_tokens_action}"
            )
            logger.error(msg)
            raise GenerationError(msg)
        record_jsonl = self._convert_records_to_jsonl(dict(records))
        tokenized = self.tokenizer(record_jsonl["text"], add_special_tokens=False)
        max_new_tokens = self.metadata.max_seq_length - len(self.schema_prompt_ids)
        # Both the prompt and the records are enclosed by special tokens.
        # TODO: This is no longer always accurate, sometimes only a bos token is
        # added to the prompt, and eventually we may experiment with multi-token
        # delimiters for each group
        max_new_tokens -= 2 * NUM_SPECIAL_TOKENS
        for ids in tokenized["input_ids"]:
            if len(ids) > max_new_tokens:
                max_tokens_action = _get_max_tokens_action(self.metadata.rope_scaling_factor)
                msg = (
                    "At least one record requires more tokens than fit in the "
                    f"available context length. {max_tokens_action}"
                )
                logger.error(msg)
                raise GenerationError(msg)
            self.stats["tokens_per_record"].update(len(ids))
        tokenized.update({"text": record_jsonl["text"]})
        return tokenized

    def _tokenize_dataset(self, dataset: Dataset, keep_columns: list[str] | None = None) -> Dataset:
        """Tokenize the records in the dataset.

        Args:
            dataset: 🤗 Dataset object to be tokenized.
            keep_columns: List of columns to keep in the dataset. This is useful if
                you need certain fields for subsequent processing (e.g., grouping).

        Returns:
            The tokenized Dataset object.
        """
        keep_columns = keep_columns or []
        logger.info("Tokenizing records")

        # Ensure the cache directory exists, which apparently is required
        # for datasets>=3 ?
        cache_dir = self.cache_file_path.parent
        if "is_val" in dataset.info.description:
            cache_file = str(self.cache_file_path.with_suffix(".val.tokens.arrow"))
        else:
            cache_file = str(self.cache_file_path.with_suffix(".tokens.arrow"))
        os.makedirs(cache_dir, exist_ok=True)

        return dataset.map(
            self._tokenize_records,
            batched=True,
            desc="Tokenizing records",
            cache_file_name=cache_file,
            remove_columns=[c for c in dataset.column_names if c not in keep_columns],
        )

    def _run_example_generation(self, generator: Callable, dataset: Dataset) -> Dataset:
        try:
            return Dataset.from_generator(
                generator=generator,
                gen_kwargs={"dataset": dataset},
                cache_dir=str(self.cache_file_path),
            )
        except DatasetGenerationError as err:
            raise GenerationError("The generator provided for dataset generation ran into errors.") from err


class TabularDataExampleAssembler(TrainingExampleAssembler):
    """Standard tabular data example assembler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def num_records_train(self) -> int:
        return len(self.train_dataset)

    @property
    def num_records_validation(self) -> int:
        return 0 if self.validation_dataset is None else len(self.validation_dataset)

    def _preprocess_before_splitting(self, tokenized_records: Dataset) -> Dataset:
        """Tabular data examples do not require any preprocessing before splitting."""
        return tokenized_records

    def _fill_context_with_records_generator(self, dataset: Dataset) -> GeneratorType:
        """Generate examples that fill the available context window with records.

        Each example consists of a prompt followed by a single sequence of records,
        which is enclosed by BOS and EOS special tokens.

        Args:
            dataset: Tokenized 🤗 Dataset to be used for example generation.
        """
        num_rows = len(dataset)
        max_new_tokens = self.metadata.max_seq_length - len(self.schema_prompt_ids)
        # Both the prompt and the records are enclosed by special tokens.
        # TODO: 2 * num_special_tokens may not be a valid assumption
        max_new_tokens -= 2 * NUM_SPECIAL_TOKENS
        num_sequences = 0

        i = 0
        num_examples = 0
        update_interval = max(1, num_rows // 20)

        for j in range(1, len(dataset) + 1):
            if j % update_interval == 0:
                logger.info(f"Assembling examples: {j}/{num_rows} records")
            num_sequences += 1
            # If
            # 1) this is the last record or
            # 2) we have already added `max_sequences_per_example` records to the example or
            # 3) adding the next record would exceed the max length
            # the next record would exceed the max length, yield the
            # example.
            if (
                j == len(dataset) - 1
                or num_sequences == self.metadata.max_sequences_per_example
                or (sum(len(ids) for ids in dataset[i : j + 1]["input_ids"]) > max_new_tokens)
            ):
                # Each example will fill the entire available context window.
                example = Example(
                    prompt=self.schema_prompt,
                    tokenizer=self.tokenizer,
                    metadata=self.metadata,
                )
                input_ids = dataset[i:j]["input_ids"]
                attention_mask = dataset[i:j]["attention_mask"]

                example.add_sequence(
                    {
                        "input_ids": list(chain(*input_ids)),
                        "attention_mask": list(chain(*attention_mask)),
                    }
                )
                # Track stats for train and val sets separately, but will only report combined stats
                if "is_val" in dataset.info.description:
                    stats = self.stats_val
                else:
                    stats = self.stats
                stats["records_per_example"].update(len(input_ids))
                stats["tokens_per_example"].update(example.num_tokens)
                yield example.to_dict()

                i = j
                num_examples += 1
                num_sequences = 0

    def _prepare_dataset_for_training(
        self, dataset: Dataset, data_fraction: float, rng: np.random.Generator
    ) -> Dataset | None:
        """Prepare a dataset for training by shuffling and potentially duplicating it.

        This function handles the preparation of both training and validation datasets. For training
        datasets, it can duplicate the data based on the data_fraction parameter. For validation
        datasets, it simply shuffles the data once.

        Args:
            dataset: The input dataset to prepare
            data_fraction: Fraction of the dataset to use. For training datasets, this can
                be > 1 to duplicate the data multiple times. For test datasets, this is
                ignored.

        Returns:
            A prepared dataset ready for training or validation. Returns None if the input
            dataset is None.
        """

        if dataset is None:
            return None

        ds_list = []
        decimal, integer = math.modf(data_fraction)

        # Build up integer part of shuffled dataset.
        if "is_val" in dataset.info.description:
            # we do not need to duplicate the dataset for test set
            ds_list.append(dataset.shuffle(generator=rng))
        else:
            for _ in range(int(integer)):
                ds_list.append(dataset.shuffle(generator=rng))

            # If there is a decimal part, add shuffled subset of the dataset.
            if decimal > 0:
                ds_list.append(dataset.shuffle(generator=rng).select(range(math.ceil(decimal * len(dataset)))))

        # Flattening indices rewrites to disk and speeds up subsequent data access.
        processed_dataset = concatenate_datasets(ds_list).flatten_indices()
        res = self._run_example_generation(self._fill_context_with_records_generator, processed_dataset)
        return res

    def assemble_training_examples(self, data_fraction: float = 1.0) -> TrainingExamples:
        """Build examples with randomly shuffled records.

        Args:
            data_fraction: Fraction of the dataset to use for example generation.

        Returns:
            TrainingExamples object containing a 🤗 Dataset objects for the train
            and test set of examples, as well as an object with associated statistics.
        """
        logger.info(
            f"Assembling examples from {data_fraction:.1%} of the input records",
        )

        rng = utils.get_random_number_generator(self.seed)
        # Process both training and test datasets
        training_dataset = self._prepare_dataset_for_training(self.train_dataset, data_fraction, rng)
        validation_dataset = self._prepare_dataset_for_training(self.validation_dataset, 1.0, rng)

        examples = TrainingExamples(
            train=training_dataset,
            test=validation_dataset,
            stats={
                "tokens_per_record": self.stats["tokens_per_record"],
                "tokens_per_example": self.stats["tokens_per_example"],
                "records_per_example": self.stats["records_per_example"],
            },
        )

        utils.log_training_example_stats(examples.stats)

        return examples


class GroupedDataExampleAssembler(TrainingExampleAssembler):
    """Grouped data example assembler.

    Args:
        group_training_examples_by: Column to group training examples by.
        order_training_examples_by: Column to order training examples by.
        dataset: Dataset to be processed.
        tokenizer: Tokenizer used for tokenizing the dataset records.
        metadata: training configuration, e.g., group by, order by, prompt
            template, bos/eos tokens and where to use them.
        test_size: Fraction of the dataset to use for testing. If None, there will
            be no test set and hence no evaluation during training.
        cache_file_path: Path to store the cached dataset for efficient data access.
        seed: Seed for the random number generator and train-test split.
    """

    def __init__(
        self,
        group_training_examples_by: str,
        order_training_examples_by: str | None,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        metadata: ModelMetadata,
        test_size: int | float | None = None,
        cache_file_path: str | Path | None = None,
        seed: int | None = None,
        keep_columns: list[str] | None = None,
        *args,
        **kwargs,
    ):
        if group_training_examples_by is None:
            raise ValueError("GroupedDataExampleAssembler created with no groupby columns set")

        self.group_by: list[str] = [group_training_examples_by]
        self.order_by: str | None = order_training_examples_by

        # GroupedDataExampleAssembler needs group_by and order_by columns for its processing.
        # Merge any caller-provided keep_columns with the required columns.
        required_columns = self.group_by.copy()
        if self.order_by is not None:
            required_columns.append(self.order_by)
        if keep_columns:
            required_columns = list(set(required_columns + keep_columns))

        # We need to split the dataset first so that the grouping column(s) are still present when we invoke
        # `utils.grouped_train_test_split`. After the split we tokenize and perform the (potentially expensive) grouping step independently for
        # train and test.
        if test_size is not None and test_size > 0:
            df_dataset = dataset.to_pandas()
            train_raw, test_raw = grouped_train_test_split(
                df_dataset,
                group_by=self.group_by[0],
                test_size=test_size,
                random_state=seed,
            )
            train_raw = Dataset.from_pandas(train_raw)
            if isinstance(test_raw, pd.DataFrame):
                test_raw = Dataset.from_pandas(test_raw)
                test_raw.info.description += "is_val"
        else:
            train_raw = dataset
            test_raw = None

        super().__init__(
            dataset=train_raw,
            tokenizer=tokenizer,
            metadata=metadata,
            keep_columns=required_columns,
            test_size=None,  # we already did the split
            cache_file_path=cache_file_path,
            seed=seed,
            *args,
            **kwargs,
        )

        # tokenize and preprocess the test set if it exists
        if test_raw is not None:
            tokenized_test = self._tokenize_dataset(test_raw, required_columns)
            processed_test = self._preprocess_before_splitting(tokenized_test)
            self.validation_dataset = processed_test
            self.validation_dataset.info.description += "is_val"
        else:
            self.validation_dataset = None

    @property
    def num_records_train(self) -> int:
        return sum(self.train_dataset["num_records"])

    @property
    def num_records_validation(self) -> int:
        return 0 if self.validation_dataset is None else sum(self.validation_dataset["num_records"])

    @property
    def num_groups_train(self) -> int:
        return len(self.train_dataset)

    @property
    def num_groups_validation(self) -> int:
        return 0 if self.validation_dataset is None else len(self.validation_dataset)

    def _preprocess_before_splitting(self, tokenized_records: Dataset) -> Dataset:
        """Group and order the tokenized records before splitting the dataset."""
        if self.order_by is not None:
            tokenized_records = self._order_records(tokenized_records, self.order_by)

        grouped_dataset = self._run_example_generation(self._group_tokenized_records_generator, tokenized_records)

        return grouped_dataset

    def _group_tokenized_records_generator(self, dataset: Dataset) -> GeneratorType:
        """Group tokenized records using the specified group_by column.

        Args:
            dataset: Tokenized 🤗 Dataset to be grouped.

        Yields:
            Dictionary with keys for 'input_ids', 'attention_mask', and
            'num_records' for each group.
        """
        for col in self.group_by:
            if col not in self.keep_columns:
                msg = f"Grouping column '{col}' not found in the `keep_columns` list."
                logger.error(msg)
                raise ParameterError(msg)

        # Pandas complains if group_by is a list of length 1.
        group_by = self.group_by if len(self.group_by) > 1 else self.group_by[0]

        logger.info(
            f"Grouping tokenized records by '{group_by}'",
        )

        grouped = dataset.select_columns(group_by).to_pandas().groupby(group_by)
        update_interval = min(100, grouped.ngroups // 10)

        for _, group_data in tqdm(
            grouped,
            total=grouped.ngroups,
            desc=f"Grouping records by '{group_by}'",
            miniters=update_interval,
        ):
            group_indices = group_data.index.to_list()
            group = dataset.select(group_indices).remove_columns(group_by).to_dict()
            num_records = len(group["input_ids"])
            self.stats["records_per_group"].update(num_records)
            group["input_ids"] = list(chain(*group["input_ids"]))
            group["attention_mask"] = list(chain(*group["attention_mask"]))
            self.stats["tokens_per_group"].update(len(group["input_ids"]))
            group.update({"num_records": num_records})
            yield group

    def _fill_context_with_groups_generator(self, dataset: Dataset) -> GeneratorType:
        """Generate examples that fill the available context window with groups.

        Each example consists of a prompt followed by multiple sequences of groups, which
        are each enclosed BOS and EOS special tokens.

        Args:
            dataset: Tokenized 🤗 Dataset to be used for example generation.
        """
        num_examples = 0
        num_sequences = 0

        num_groups = len(dataset)
        max_new_tokens = self.metadata.max_seq_length - len(self.schema_prompt_ids)
        # The prompt is enclosed by special tokens.
        max_new_tokens -= NUM_SPECIAL_TOKENS
        update_interval = min(100, num_groups // 10)
        # Create example for the first group.
        num_records = 0
        example = Example(
            prompt=self.schema_prompt,
            tokenizer=self.tokenizer,
            metadata=self.metadata,
        )

        for i in range(len(dataset)):
            if i % update_interval == 0:
                logger.info(f"Assembling examples: {i}/{num_groups} groups")
            num_sequences += 1
            example.add_sequence(dataset[i])
            num_records += dataset[i]["num_records"]
            # If 1) this is the last group or 2) we have already added
            # max_sequences_per_example groups to the example or 3) adding
            # the next group would exceed the max length, yield the
            # example.
            if (
                i == len(dataset) - 1
                or num_sequences == self.metadata.max_sequences_per_example
                or (
                    # TODO: is this accurate in including special tokens properly?
                    example.num_tokens + len(dataset[i + 1]["input_ids"]) > max_new_tokens
                )
            ):
                yield example.to_dict()

                num_examples += 1
                num_sequences = 0

                # Track stats for train and val sets separately, but will only report combined stats
                if "is_val" in dataset.info.description:
                    stats = self.stats_val
                else:
                    stats = self.stats
                stats["groups_per_example"].update(example.num_sequences)
                stats["tokens_per_example"].update(example.num_tokens)
                stats["records_per_example"].update(num_records)

                # Create new example for the next group.
                num_records = 0
                example = Example(
                    prompt=self.schema_prompt,
                    tokenizer=self.tokenizer,
                    metadata=self.metadata,
                )

    def _prepare_dataset_for_training(
        self, dataset: Dataset, data_fraction: float, rng: np.random.Generator
    ) -> Dataset | None:
        """Prepare a dataset for training by shuffling and potentially duplicating it.

        This function handles the preparation of both training and validation datasets. For training
        datasets, it can duplicate the data based on the data_fraction parameter. For validation
        datasets, it simply shuffles the data once.

        Args:
            dataset: The input dataset to prepare
            data_fraction: Fraction of the dataset to use. For training datasets, this can
                be > 1 to duplicate the data multiple times. For test datasets, this is
                ignored.

        Returns:
            A prepared dataset ready for training or validation. Returns None if the input
            dataset is None.
        """
        if dataset is None:
            return None

        ds_list = []
        decimal, integer = math.modf(data_fraction)

        if "is_val" in dataset.info.description:
            # we do not need to duplicate the dataset for test set
            ds_list.append(dataset.shuffle(generator=rng))
        else:
            # Build up integer part of grouped dataset.
            # Each integer part is composed of all the groups.
            for _ in range(int(integer)):
                ds_list.append(dataset.shuffle(generator=rng))

            # If there is a decimal part, add a subset of the grouped dataset.
            # We add groups until the cumulative sum of records is greater than or equal
            # to the number of records needed.
            if decimal > 0:
                num_records_needed = math.ceil(decimal * self.num_records_train)
                group_shuffled = dataset.shuffle(generator=rng)
                records_cumsum = np.cumsum(group_shuffled["num_records"])
                # The following condition (`records_cumsum >= num_records_needed`) will always have at least one truthy value:
                # - For small decimal values (e.g., 0.01), the first index of `records_cumsum` usually meets the condition.
                # - For large decimal values (e.g., 0.99), the last index of `records_cumsum` meets the condition because:
                # `records_cumsum[-1] == self.num_records_train`, which is always >= `num_records_needed`.
                where = np.argwhere(records_cumsum >= num_records_needed).flatten()[0]
                # `where` represents the index of the last group to be included, based on `num_records_needed`.
                # We add 1 to the range because indexing starts at 0.
                ds_list.append(group_shuffled.select(range(where + 1)))
        # Flattening indices rewrites to disk and speeds up subsequent data access.
        processed_dataset = concatenate_datasets(ds_list).flatten_indices()

        return self._run_example_generation(self._fill_context_with_groups_generator, processed_dataset)

    def assemble_training_examples(self, data_fraction: float = 1.0) -> TrainingExamples:
        """Build examples with grouped (and optionally ordered) records.

        Args:
            data_fraction: Fraction of the dataset to use for example generation.

        Returns:
            TrainingExamples object containing a 🤗 Dataset objects for the train
            and test set of examples, as well as an object with associated statistics.
        """
        logger.info(
            f"Assembling grouped examples from {data_fraction:.1%} of the input training records",
        )

        rng = utils.get_random_number_generator(self.seed)
        # Process both training and validation datasets
        training_dataset = self._prepare_dataset_for_training(self.train_dataset, data_fraction, rng)
        validation_dataset = self._prepare_dataset_for_training(self.validation_dataset, 1.0, rng)

        examples = TrainingExamples(
            train=training_dataset,
            test=validation_dataset,
            stats={
                "tokens_per_record": self.stats["tokens_per_record"],
                "tokens_per_group": self.stats["tokens_per_group"],
                "tokens_per_example": self.stats["tokens_per_example"],
                "records_per_example": self.stats["records_per_example"],
                "groups_per_example": self.stats["groups_per_example"],
            },
        )

        utils.log_training_example_stats(examples.stats)

        return examples
