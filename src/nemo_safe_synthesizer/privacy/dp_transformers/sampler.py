# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/src/dp_transformers/sampler.py
# See THIRD_PARTY.md for the original MIT license terms.
from typing import Iterator, Sequence

import torch
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler

from nemo_safe_synthesizer.observability import get_logger

logger = get_logger(__name__)


class _EntitySampler(Sampler):
    def __init__(self, entity_sampler: Sampler, entity_mapping: Sequence[Sequence[int]]):
        self.entity_mapping = list(entity_mapping)
        self.entity_sampler = entity_sampler
        self.indices = [0] * len(self.entity_mapping)

    def __len__(self) -> int:
        return len(self.entity_sampler)

    def __iter__(self) -> Iterator[list[int]]:
        # each batch from the entity_sampler gives a set of entity_ids.
        for batch_entity_ids in self.entity_sampler:
            sample_ids = [self.indices[entity_id] for entity_id in batch_entity_ids]
            # For each entity_id, we sample one element from the entity.
            for entity_id in batch_entity_ids:
                # First track which index in the entity_mapping to sample next
                self.indices[entity_id] += 1
                # Then ensure that the updated index does not exceed the total
                # number of samples aviailable for the entity. If so, we cycle
                # back to the first sample available for the entity.
                self.indices[entity_id] = self.indices[entity_id] % len(self.entity_mapping[entity_id])
            yield [
                int(self.entity_mapping[entity_id][sample_id])
                for entity_id, sample_id in zip(batch_entity_ids, sample_ids)
            ]


class ShuffledEntitySampler(_EntitySampler):
    def __init__(self, entity_mapping: Sequence[Sequence[int]], batch_size: int) -> None:
        """
        This Sampler class uses
            1. RandomSampler to shuffle the entities in dataset, and select
               `batch_size` number of entities from the dataset.
            2. BatchSampler to sample one element from an entity in a given
               batch.
        Each batch gets one sample from any entity in the dataset. This ensures
        that any training step isn't overly influenced by one entity's data
        (particularly important when training for less than one epoch).

        Args:
            entity_mapping: A mapping where `dataset[entity_mapping[i][j]]`
                produces the j-th sample of the i-th entity in the dataset.
            batch_size: Number of examples included in each batch.
        """

        entity_sampler = BatchSampler(RandomSampler(entity_mapping), batch_size=batch_size, drop_last=True)
        super().__init__(entity_sampler, entity_mapping)


class PoissonEntitySampler(_EntitySampler):
    def __init__(self, entity_mapping: Sequence[Sequence[int]], sample_rate: float) -> None:
        """
        This Sampler class uses Poisson sampling, i.e., entities in the
        dataset are added to the batch with a given probability
        `sample_rate`. Note that, because of that, batch size is not always
        the same, but in average equal to `len(dataset) * sample_rate`.
        Empty batches are skipped.

        Args:
            entity_mapping: A mapping where `dataset[entity_mapping[i][j]]`
                produces the j-th sample of the i-th entity in the dataset.
            sample_rate: Probability of any given entity being included in
                the batch.
        """

        entity_sampler = UniformWithReplacementNonNullSampler(
            num_samples=len(entity_mapping),
            sample_rate=sample_rate,
        )
        super().__init__(entity_sampler, entity_mapping)


class UniformWithReplacementNonNullSampler(UniformWithReplacementSampler):
    """This sampler is similar to the `UniformWithReplacementSampler`, but it
    never outputs empty batches, while ensuring they are taken into account."""

    def __init__(self, *args, **kwargs):
        # NOTE: we might want to log empty_batches for debugging purposes
        self.empty_batches = 0
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.steps - self.empty_batches

    def __iter__(self):
        self.empty_batches = 0
        num_batches = self.steps
        while num_batches > 0:
            mask = torch.rand(self.num_samples, generator=self.generator) < self.sample_rate
            indices = mask.nonzero(as_tuple=False).reshape(-1).tolist()
            if len(indices) > 0:  # Don't output batch if it's empty ...
                logger.debug(f"Samples used: {indices}")
                yield indices
            else:
                self.empty_batches += 1

            # ... but make sure to count it
            num_batches -= 1
