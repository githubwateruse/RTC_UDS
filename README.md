# Toward efficient training of neural-network control policies for real-time control of urban drainage systems

This repository accompanies the manuscript:

**Toward efficient training of neural-network control policies for real-time control of urban drainage systems.**

The repository includes the training scripts for neural-network control policies using synthetic, segmented, and continual-learning strategies, the selected trained policy parameters, and the testing scripts for the selected policies.

## Repository contents

* `SWMM_Astlingen/`
* `policy_training/`
* `trained_policies/`
* `policy_testing/`

## Requirements

The original experiments were conducted using Python 3.8.0. The main Python dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```text
# Python 3.8.0
numpy==1.24.4
pyswmm==1.4.0
xlrd==1.2.0
XlsxWriter==3.2.9
```

The package versions reflect the environment used for the core implementation scripts. Minor version adjustments may be required depending on the local SWMM, PySWMM, and operating-system configuration.
The scripts are research implementation programs rather than a packaged software library. 

## Usage

The training and testing scripts are interactive; select the configuration when prompted.

```bash
cd policy_training
python train_baselines.py            # synthetic and segmented baselines
python train_continue_learning.py    # segmented continual learning
```

```bash
cd policy_testing
python run_testing.py                # test the selected policies
```

Training outputs are written to `policy_training/training_outputs/`, and testing outputs to `policy_testing/testing_outputs/`.

## Selected policies

The repository provides the selected neural-network control policies used in the study. For each training configuration, the best-performing saved generation was selected based on validation performance.

| Experiment | Policy file |
|---|---|
| Synthetic baseline, 5 min | `policy_synthetic_baseline_5min.xlsx` |
| Synthetic baseline, 15 min | `policy_synthetic_baseline_15min.xlsx` |
| Segmented baseline, 2000, 5 min | `policy_segmented_baseline_2000_5min.xlsx` |
| Segmented baseline, 2000, 15 min | `policy_segmented_baseline_2000_15min.xlsx` |
| Segmented baseline, 2000, 1 h | `policy_segmented_baseline_2000_1h.xlsx` |
| Segmented baseline, 2000, 6 h | `policy_segmented_baseline_2000_6h.xlsx` |
| Segmented continual learning, 2000, 5 min | `policy_segmented_continue_learning_2000_5min.xlsx` |
| Segmented continual learning, 2000, 15 min | `policy_segmented_continue_learning_2000_15min.xlsx` |
| Segmented baseline, 2000–2004, 5 min | `policy_segmented_baseline_2000_2004_5min.xlsx` |
| Segmented baseline, 2000–2004, 15 min | `policy_segmented_baseline_2000_2004_15min.xlsx` |
| Segmented continual learning, 2000–2004, 5 min | `policy_segmented_continue_learning_2000_2004_5min.xlsx` |
| Segmented continual learning, 2000–2004, 15 min | `policy_segmented_continue_learning_2000_2004_15min.xlsx` |

The selected policy files are provided directly in `trained_policies/`, so users can run the testing scripts without retraining the neural-network policies from scratch.


## Acknowledgements

This repository uses the Astlingen SWMM benchmark model introduced by Sun et al. (2020):

> Sun, C., Svensen, J. L., Borup, M., Puig, V., Cembrano, G., and Vezzaro, L. (2020). *An MPC-Enabled SWMM Implementation of the Astlingen RTC Benchmarking Network*. Water, 12(4), 1034. https://doi.org/10.3390/w12041034

The Astlingen SWMM model files are available from the open-toolbox repository: https://github.com/open-toolbox/SWMM-Astlingen

We sincerely acknowledge the authors for developing and publicly providing the Astlingen SWMM benchmark model.
