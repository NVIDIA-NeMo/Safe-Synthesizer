# NSS Experiment Guide

This guide outlines the standard approaches for running experiments across different models in NSS. Detailed instructions for running NSS Slurm jobs can be found in the [dedicated README](README.md).

---

## Unsloth Configuration Experiments

### Parameter Setup

- **Number of generated records (`num_records`)**: Set to 1000.  
  This value was chosen to align with current NSS job runs, which use the default 1k records. This allows for a more accurate comparison of total runtime. Going forward, we plan to increase this value to obtain more stable scores across runs.

- **Learning rate (LR)**:
  - 0.0001 for Mistral
  - 0.0005 for SmolLM  
  (A PR has been opened to automate this condition.)

- **`max_sequences_per_example`**: Set to 10.  
  (Another PR has been opened to automate this condition as well.)

### Cluster configuration

- Most experiments were conducted in the **ORD cluster** using 10 A100 GPUs.
- A small number of experiments were also run in the **DFW cluster** using H100 GPUs, primarily to compare runtime performance against A100.

### Weights & Biases (W&B)

- Enabled for all experiments by setting:
  ```bash
  WANDB_MODE="online"
  ```

---

## Post-Experiment Analysis

All experiment results were collected using Weights & Biases (wandb). The main metrics used to evaluate and select candidate models were:

- Success rate across jobs within each experiment
- Synthetic Quality Score
- Total runtime (reported as `eval/total_time_sec` in wandb)
- Data Privacy Score
- Valid Record Fraction

---

## DP Configuration Experiments (TODO)

- **Batch size**: 8
- **Structured generation backend**: TBD
