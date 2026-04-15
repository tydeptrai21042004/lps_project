# LPS-TCN comparison project

This project is now aligned to the paper-facing **single-stage `lps_conv` proposal**. The comparison code keeps the same paper baselines, but removes the extra internal proposal-family variants from the public benchmark flow so the reported results stay focused on the method described in the manuscript.

## Included datasets

### Vision-as-sequence datasets
- `seqmnist`
- `fashion_mnist`
- `kmnist`
- `emnist_digits`
- `cifar10_gray`

### Time-series archive datasets
- `arrowhead`
- `basicmotions`
- `coffee`
- `earthquakes`
- `ecg200`
- `ecg5000`
- `electricdevices`
- `facefour`
- `forda`
- `gunpoint`
- `italy_power_demand`
- `two_patterns`
- `wafer`

### Built-in synthetic stress datasets
- `synthetic_sines`
- `synthetic_shiftmix`
- `synthetic_multiscale`

## Included models

### TCN-family baselines
- `tcn_plain`
- `smoothed_tcn`
- `gaussian_tcn`
- `hamming_tcn`
- `savgol_tcn`
- `moving_avg_tcn`
- `learnable_front_tcn`

### Paper proposal
- `lps_conv`

### Non-TCN baselines
- `lstm`
- `bilstm`
- `gru`
- `bigru`
- `fcn`

## What changed

- real time-series dataset path via aeon archive loaders
- built-in synthetic stress datasets for shift/noise robustness checks
- train-statistics standardization for time-series data
- stronger RNN baselines with mean/max/attention pooling and optional input projection
- stronger FCN baseline
- unconstrained learnable front-end baseline (`learnable_front_tcn`)
- comparison runner now keeps going when one model fails
- comparison runner supports multi-dataset evaluation and predefined fair dataset groups
- public paper comparison flow now keeps only the single-stage `lps_conv` proposal

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Train the paper proposal on ECG5000:

```bash
python train.py   --dataset ecg5000   --model lps_conv   --output-dir ./outputs/lps_conv_ecg5000
```

Run a broader comparison on one dataset:

```bash
python compare_models.py   --dataset ecg5000   --model-set paper_compare   --output-dir ./outputs/compare_ecg5000
```

List all supported datasets and predefined dataset groups:

```bash
python compare_models.py --list-datasets
```

Run a fairer multi-dataset benchmark using a predefined archive group:

```bash
python compare_models.py   --dataset-set fair_ucr_core   --model-set paper_compare   --output-dir ./outputs/compare_fair_ucr_core
```

Run a custom multi-dataset benchmark:

```bash
python compare_models.py   --datasets ecg200,ecg5000,gunpoint,italy_power_demand   --model-set paper_compare   --output-dir ./outputs/compare_custom_ucr
```

## Paper-supported comparison sets

The comparison runner distinguishes between:

- `paper_compare`: only paper-supported baselines plus the single-stage proposal.
- `paper_baselines`: only literature-backed baselines.
- `ablations`: internal engineering variants and fixed-smoother ablations useful for diagnostics but not for the main paper table.

Paper-backed TCN-side baselines:

- `hybrid_dilated_tcn`: HDC-inspired non-gridding dilation schedule baseline.
- `smoothed_tcn`: smoothed dilated convolution baseline.
- `blurpool_tcn`: fixed anti-aliased blurpool-style low-pass front-end baseline.

A provenance manifest is saved to `model_provenance.json` for each comparison run.

## Recommended fair test sets

- `quick_archive`: fast smoke test on `ecg200`, `ecg5000`, `gunpoint`
- `fair_ucr_core`: balanced core set for paper-style checks on `ecg200`, `ecg5000`, `gunpoint`, `italy_power_demand`, `forda`, `wafer`, `electricdevices`, `two_patterns`
- `fair_ucr_extended`: adds smaller classic datasets such as `arrowhead`, `coffee`, `earthquakes`, and `facefour`
- `fair_multivariate`: `basicmotions`
- `fair_all`: union of the archive datasets above

When you run more than one dataset, the comparison runner saves:

- `per_run_results.csv`
- `aggregate_results.csv`
- `aggregate_results_by_dataset.csv`
- `macro_results_across_datasets.csv`

The macro table is the fairest one to quote when different datasets have very different difficulty levels.
