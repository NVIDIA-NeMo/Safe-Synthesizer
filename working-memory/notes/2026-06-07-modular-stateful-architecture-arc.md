---
title: Architecture arc — self-managing components under an async state-machine orchestrator
date: 2026-06-07
status: direction
kb:
  tags: [architecture, design-arc, state-machine, orchestrator, config, backends, testability, decoupling]
  summary: >
    Multi-month direction (from the operator, 2026-06-07): grow more backend-like,
    self-contained components that expose knobs slowly and decouple, coordinated by an
    async orchestrator that treats each component — and each component treats itself — as
    a state machine over a single shared state system. Config becomes first-class
    throughout. Backends and config are the seed; evaluation, assembly, and PII follow.
---

# Architecture arc: self-managing components + async state-machine orchestrator

Direction set by the operator on 2026-06-07. This is a horizon note, not a plan — it
records the intended shape so future work bends toward it. Related session work:
[[2026-06-07-remote-generation-backend]].

## The vision (distilled)

- More **backend-like components**: modular, well-contained, with knobs exposed *slowly*
  and dependencies *decoupled* over time — not a big-bang redesign.
- An **async orchestrator** in the middle that treats each component as a **state machine**;
  each component also **manages itself** as a state machine (same model at both levels —
  fractal/self-similar).
- **Configuration is first-class** throughout the code: a component's config is its
  declarative spec, and the orchestrator composes components from config.
- A **single state-management system** means relocating a task from one place to another
  (process, host, remote endpoint) is routine rather than special-cased.
- Continue reinforcing **performance and testability** of each stage.
- The **config layer and the backends** already embody the well-designed version of this;
  **evaluation, assembly, and PII** are the parts that should converge to it next.

## What the codebase already embodies

- **Typed, nested config** (`config/`, `SafeSynthesizerParameters`): the proto "config as
  first-class spec." Pydantic models + validators are already the declarative substrate.
- **`GeneratorBackend` lifecycle** (`initialize → prepare_params → generate → teardown`) is
  a **proto state machine**: explicit phases, idempotent teardown, a `_torn_down` guard.
  The template-method refactor this session made the *shared* transitions concrete on the
  base and pushed only engine-specific transitions to subclasses — a step toward a uniform
  component contract.
- **`RemoteBackend`** is a proof point for "relocate a task elsewhere": the *same* generate
  loop runs whether the engine is in-process (vLLM) or on a remote endpoint (HTTP). The
  state lived in the loop, not the engine, so the engine became swappable. That is exactly
  the payoff the single-state-system vision predicts.
- **Resume overlay** (`_overlay_set_fields` in `config/parameters.py`) already treats a run
  as resumable state reconstructed from config — a primitive form of durable orchestration.

## Target model (how the pieces fit)

Three contracts, applied uniformly:

1. **Component contract** = `typed config` + `explicit lifecycle/state` + `decoupled I/O`.
   The I/O decoupling is the load-bearing one: the batch loop's "text in / records out"
   boundary is why generation became testable and relocatable. Every stage should aim for
   an equally narrow, engine-free data boundary.
2. **Two component archetypes** (don't force one shape on both):
   - **Engines** — resource-owning, long-lived, multi-phase: `training`, `generation`.
     The full FSM fits naturally (acquire → configure → run → release, with self-managed
     internal transitions and guards).
   - **Transforms** — mostly pure stage functions: `assembly`, `pii_replacer`, much of
     `evaluation`. These are *degenerate* state machines (effectively one transition).
     Wrap them in the same contract so the orchestrator is uniform, but resist inventing
     phases they don't have. The win for transforms is the config + I/O contract and
     testability, not elaborate state.
3. **Async orchestrator** — drives components as states over a **single state/task store**.
   Today the orchestrator is the synchronous SDK builder (`SafeSynthesizer.run` →
   train → generate → evaluate). The arc replaces that with an async coordinator where
   transitions are explicit and state is durable, which buys, roughly for free:
   - **Concurrency** (independent stages overlap),
   - **Resumability** (re-enter from the last committed state — generalizes the resume overlay),
   - **Relocation** (a stage can run local, in another process, or remote — RemoteBackend
     is the first instance),
   - **Observability** (the existing `traced`/`CategoryLogger` maps onto state transitions).

## Why it pays off

- **Testability**: pure stages with narrow data boundaries are unit-testable without GPUs
  or servers (the 24 GPU-free `RemoteBackend` tests are the template).
- **Performance**: concurrency + relocation (offload generation to a fleet of endpoints;
  parallelize independent eval components) without rewriting stage internals.
- **Evolvability**: "expose more knobs slowly" = additive config fields behind a stable
  seam. The seam (abstract contract) is what lets knobs grow without churn downstream.

## Tensions / risks to watch (assumptions that could bite)

- **Premature FSM formalism.** A heavyweight state-machine framework imposed on transforms
  that have one transition is pure tax. Introduce the orchestrator where statefulness is
  real (engines) first; let transforms stay near-pure.
- **The state store becomes the new coupling point.** A single shared state system is
  leverage *and* a chokepoint — its schema/versioning and failure semantics need the same
  rigor the config layer got. Get the serialization/resume contract right early.
- **Config sprawl.** "Config first-class" can drift into a thousand flat knobs. Keep config
  *nested by component* (the current pattern) and keep cross-component invariants in
  model-validators (e.g. the remote+time-series check) so the orchestrator stays dumb.
- **Uniform error/observability model is a prerequisite, not an afterthought.** Components
  as states only compose if they fail and report uniformly. The `errors.py` hierarchy and
  `observability` are the seams to standardize on before multiplying components.
- **Don't decouple speculatively.** Expose a knob when there's a second consumer (RemoteBackend
  was the second consumer that justified lifting the batch loop). Decoupling without a
  driver adds indirection without payoff.

## Incremental adoption path (no big bang)

For each component, in this order, do the cheap contract work *before* handing it to an orchestrator:

1. **Backends** (mostly there): finish making the lifecycle a first-class, documented contract;
   ensure `training` and `generation` share the same phase vocabulary.
2. **Evaluation**: give each eval component (privacy, MI, AIA, PII replay) a typed config +
   a `run(inputs) -> typed result` boundary; they're already component-shaped in `evaluation/`.
   These parallelize trivially once decoupled — an early performance win.
3. **Assembly / data_processing**: express as transforms with explicit config (token budget,
   validators are already factored out — `budget.py`, `validation.py`).
4. **PII replacer**: transform contract; the NER step is the resource-owning sub-engine inside it.
5. Only then introduce the **async orchestrator** over components that already satisfy the
   contract, migrating one stage at a time and keeping the synchronous builder working until
   parity is proven.

## Open questions

- What is the state/task store? (In-memory + journal? A real durable store? It must satisfy
  the resume + relocation goals without coupling stage internals.)
- Engine vs transform: is a two-tier contract worth the explicitness, or does one contract
  with optional phases suffice? (Lean: one contract, phases optional, transforms ignore them.)
- How far does "self-managing component" go — do components own retries/backoff (like
  RemoteBackend's per-request behavior) or does the orchestrator? (Lean: component owns
  intra-stage resilience; orchestrator owns inter-stage transitions.)
- Where does the orchestrator live relative to the SDK builder and the CLI? (The builder is
  the natural incubation site; keep it as the stable façade.)
