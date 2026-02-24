# NSS Experiment Guide

This guide outlines the standard approaches for running experiments across different models in NSS. Detailed instructions for running NSS Slurm jobs can be found in the [dedicated README](README.md).

---

## Configuration Experiments

### Parameter Setup

- **Number of generated records (`num_records`)**: Set to 1000.  
  This value was chosen to align with current NSS job runs, which use the default 1k records. This allows for a more accurate comparison of total runtime. Going forward, we plan to increase this value to obtain more stable scores across runs.
- **Number of job runs (`runs` in submit_slurm_jobs.sh)**: 
  5 runs per dataset x config combination was performed for more consistant results.  
### Cluster configuration

#### *Note: This subsection is for Nvidia's internal use only*.
- Experiments were conducted in the **ORD cluster** using A100 GPUs by default.
- We can use DFW cluster using H100 GPUs to benchmark and evaluate runtime performance.

### Weights & Biases (W&B)

- Enabled for all experiments by setting:
  ```bash
  WANDB_MODE="online"
  ```

---

## Post-Experiment Analysis

All experiment results were collected using Weights & Biases (wandb). Following main metrics were pinned in wandb, and the results were downloaded for further evaluation and candidate model selection.

- Success rate across jobs within each experiment
- Synthetic Quality Score (SQS)
- Total runtime (reported as `eval/total_time_sec` in wandb)
- Data Privacy Score (DPS)
- Valid Record Fraction


### How do we choose a candidate model (alternative) over another (control)?

We statistically analyse the experiment results using the selected metrics discussed above [from this repository](https://gitlab-master.nvidia.com/sdg-research/nss-experimental-analysis/-/tree/main). The main steps for this analysis include: 
1.  `experiment_results/clean_wandb_downloads.ipynb`: Includes metrics name mapping and classify job's states. 
2. `analysis.ipynb`: Performs statistical testing for choosing the candidate model over the default model.

*Note:* Depending on the purpose of the experiment:

- superiority stochastic test is performed: If we want to make improvement over the current implementation
- Non-inferiority stochastic test is performed: if we need to find a suitable replacement to a current model for reasons other than performance, eg. runtime, security etc.

We run a *p-test* for *success rate* across jobs with delta = 0.01 and a *t-test* for *SQS and DPS* with delta = 0.1 and alpha = 0.025:
  - When p-value < alpha → reject default model in favor of the candidate model 

Total runtime was another key factor in our evaluation. We primarily considered the average runtime across all jobs and, based on overall performance, selected an option that provides a balanced outcome.

---
