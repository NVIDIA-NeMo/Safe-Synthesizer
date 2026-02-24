# NSS Experiment Guide

This guide outlines the standard approaches for running experiments across different models in NSS. Detailed instructions for running NSS Slurm jobs can be found in the [dedicated README](README.md).

---

## Configuration Experiments

### Parameter Setup

- **Number of generated records (`num_records`)**: Set to 1000.  
  This value was chosen to align with current NSS job runs, which use the default 1k records. This allows for a more accurate comparison of total runtime. Going forward, we plan to increase this value to obtain more stable scores across runs.
- **Number of job runs (`runs` in submit_slurm_jobs.sh)**: set to 5.
  5 runs per dataset x config combination was performed for more consistant results.  
### Cluster configuration

#### Note: This subsection is for Nvidia's internal use only.
- Most experiments were conducted in the **ORD cluster** using A100 GPUs.
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
- Synthetic Quality Score (SQS)
- Total runtime (reported as `eval/total_time_sec` in wandb)
- Data Privacy Score (DPS)
- Valid Record Fraction


*How do we select a model (alternative) over another (control)?*

- We run a *non-inferiority p-test* for *success rate* across jobs with alpha = 0.025 and
delta = 0.01:

When p-value < alpha → reject default model in favor of the candidate model → conclude the failure rate of the candidate model is not worse than control by more than delta.

- We also run a *non-inferiority t-test* for *SQS and DPS* with alpha = 0.025 and
delta = 0.1:

When p-value < alpha → reject default model in favor of the candidate model → conclude the SQS and DPS of the candidate model is not worse than control by more than delta.

- Total runtime was another key factor in our evaluation. We primarily considered the average runtime across all jobs and, based on overall performance, selected an option that provides a balanced outcome.

---
