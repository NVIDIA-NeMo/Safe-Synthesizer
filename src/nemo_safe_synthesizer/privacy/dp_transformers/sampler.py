# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/src/dp_transformers/sampler.py
# See THIRD_PARTY.md for the original MIT license terms.

"""Samplers for DP batch creation.

Provides entity-level and record-level samplers: ``ShuffledEntitySampler``
(shuffle entities, fixed batch size), ``PoissonEntitySampler`` (Poisson
sampling for proper DP accounting), and ``UniformWithReplacementNonNullSampler``
(no empty batches).
"""

from typing import Iterator, Sequence

import torch
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler

from ...observability import get_logger

logger = get_logger(__name__)


class _EntitySampler(Sampler):
    """Base sampler that maps entity-level batches to per-entity sample indices.

    Each iteration yields one sample per entity in the batch, cycling through
    each entity's samples. Used by ShuffledEntitySampler and PoissonEntitySampler.

    Args:
        entity_sampler: Sampler that yields batches of entity IDs.
        entity_mapping: For entity i, entity_mapping[i] is the list of dataset
            indices belonging to that entity. So dataset[entity_mapping[i][j]]
            is the j-th sample of entity i.
    """

    def __init__(self, entity_sampler: Sampler, entity_mapping: Sequence[Sequence[int]]):
        self.entity_mapping = list(entity_mapping)
        self.entity_sampler = entity_sampler
        self.indices = [0] * len(self.entity_mapping)

    def __len__(self) -> int:
        return len(self.entity_sampler)

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches of dataset indices, one sample per entity per batch.

        Yields:
            List of dataset indices for the current batch (one index per entity
            in the batch, cycling through each entity's samples).
        """
        # each batch from the entity_sampler gives a set of entity_ids.
        for batch_entity_ids in self.entity_sampler:
            sample_ids = [self.indices[entity_id] for entity_id in batch_entity_ids]
            # For each entity_id, we sample one element from the entity.
            for entity_id in batch_entity_ids:
                # First track which index in the entity_mapping to sample next
                self.indices[entity_id] += 1
                # Then ensure that the updated index does not exceed the total
                # number of samples available for the entity. If so, we cycle
                # back to the first sample available for the entity.
                self.indices[entity_id] = self.indices[entity_id] % len(self.entity_mapping[entity_id])
            yield [
                int(self.entity_mapping[entity_id][sample_id])
                for entity_id, sample_id in zip(batch_entity_ids, sample_ids)
            ]


class ShuffledEntitySampler(_EntitySampler):
    """Sample batches of entities at random, one sample per entity per batch.

    Uses RandomSampler to shuffle entities and BatchSampler to form batches of
    ``batch_size`` entities. Each batch contains one sample from each of the
    chosen entities, so no single entity dominates a step (important for
    entity-level DP and when training for less than one epoch).

    Args:
        entity_mapping: For entity i, entity_mapping[i] is the list of dataset
            indices for that entity; dataset[entity_mapping[i][j]] is the j-th
            sample of entity i.
        batch_size: Number of entities (and thus samples) per batch.
    """

    def __init__(self, entity_mapping: Sequence[Sequence[int]], batch_size: int) -> None:
        entity_sampler = BatchSampler(RandomSampler(entity_mapping), batch_size=batch_size, drop_last=True)
        super().__init__(entity_sampler, entity_mapping)


class PoissonEntitySampler(_EntitySampler):
    """Sample entities with Poisson (per-entity) sampling for correct DP accounting.

    Each entity is included in a batch with probability ``sample_rate``.
    Batch size varies; on average equals ``len(entities) * sample_rate``.
    Empty batches are skipped but counted toward the step budget.

    Args:
        entity_mapping: For entity i, entity_mapping[i] is the list of dataset
            indices for that entity.
        sample_rate: Probability of each entity being included in a batch.
    """

    def __init__(self, entity_mapping: Sequence[Sequence[int]], sample_rate: float) -> None:
        entity_sampler = UniformWithReplacementNonNullSampler(
            num_samples=len(entity_mapping),
            sample_rate=sample_rate,
        )
        super().__init__(entity_sampler, entity_mapping)


class UniformWithReplacementNonNullSampler(UniformWithReplacementSampler):
    """Uniform-with-replacement sampler that skips empty batches but counts them.

    Same as Opacus ``UniformWithReplacementSampler`` except batches with zero
    samples are not yielded. Empty batches are still counted toward the total
    number of steps so that step-based privacy accounting (e.g. ε composition)
    remains correct. Used by ``PoissonEntitySampler`` for Poisson sampling.

    Attributes:
        empty_batches: Number of empty batches skipped so far (reset at the
            start of each ``__iter__``; only meaningful during iteration).
    """

    def __init__(self, *args, **kwargs):
        # NOTE: we might want to log empty_batches for debugging purposes
        self.empty_batches = 0
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        """Return the number of batches that will be yielded (non-empty only).

        Equal to ``steps - empty_batches`` after a full iteration. Before
        iteration, ``empty_batches`` is 0 so the value may increase as
        empty batches are skipped during iteration.
        """
        return self.steps - self.empty_batches

    def __iter__(self) -> Iterator[list[int]]:
        """Iterate over batches, each drawn uniformly with replacement; skip empty batches.

        Each batch is formed by including each of the ``num_samples`` elements
        with probability ``sample_rate``. Batches with no samples are not
        yielded but are counted in ``empty_batches`` so privacy accounting
        stays correct.

        Yields:
            List of indices for the current batch (non-empty). Length varies
            by batch; empty batches are skipped.
        """
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

            # ... but make sure to count it toward steps
            num_batches -= 1
