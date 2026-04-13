# LPS-TCN comparison project

This project now supports both the original vision-as-sequence benchmarks and a stronger time-series path with improved baselines and an enhanced proposal model.

## Included datasets

### Vision-as-sequence datasets
- `seqmnist`
- `fashion_mnist`
- `kmnist`
- `emnist_digits`
- `cifar10_gray`

### Time-series archive datasets
- `ecg5000`
- `electricdevices`
- `forda`
- `wafer`
- `two_patterns`
- `basicmotions`

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

### Learnable front-end models
- `lps_conv`
- `lps_conv_plus`
- `lps_conv_plus_ms`

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
- enhanced proposal variant with multiscale symmetric branches, channel attention, and per-channel residual gating (`lps_conv_plus_ms`)
- more unit tests covering datasets, dtypes, pooling, and new models
- comparison runner now keeps going when one model fails

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Train the enhanced proposal on ECG5000:

```bash
python train.py   --dataset ecg5000   --model lps_conv_plus_ms   --output-dir ./outputs/lps_conv_plus_ms_ecg5000
```

Run a broader comparison:

```bash
python compare_models.py   --dataset ecg5000   --models tcn_plain,learnable_front_tcn,bilstm,bigru,fcn,lps_conv_plus,lps_conv_plus_ms   --output-dir ./outputs/compare_ecg5000
```


## Paper-supported comparison sets

The comparison runner now distinguishes between:

- `paper_compare`: only paper-supported baselines plus the proposal models.
- `paper_baselines`: only literature-backed baselines.
- `ablations`: internal engineering variants and fixed-smoother ablations that are useful for diagnostics but should not be reported as primary baselines.

New paper-backed TCN-side baselines:

- `hybrid_dilated_tcn`: HDC-inspired non-gridding dilation schedule baseline.
- `smoothed_tcn`: smoothed dilated convolution baseline.
- `blurpool_tcn`: fixed anti-aliased blurpool-style low-pass front-end baseline.

A provenance manifest is saved to `model_provenance.json` for each comparison run.
