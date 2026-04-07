# LPS-TCN comparison project

This project refactors the original single-file experiment into a small research-style codebase with stronger baselines and cleaner evaluation.

## Included models

### TCN-family baselines
- `tcn_plain`: plain dilated TCN baseline.
- `smoothed_tcn`: TCN with fixed local smoothing before each dilated convolution.
- `gaussian_tcn`: fixed Gaussian smoother + TCN.
- `hamming_tcn`: fixed Hamming smoother + TCN.
- `savgol_tcn`: fixed Savitzky-Golay smoother + TCN.
- `moving_avg_tcn`: fixed moving-average smoother + TCN.

### Learnable front-end models
- `lps_conv`: single learnable symmetric linear-phase front end + TCN.
- `lps_conv_plus`: two-stage learnable symmetric front end + TCN.

### Non-TCN sequence/classification baselines
- `lstm`: unidirectional LSTM classifier.
- `bilstm`: bidirectional LSTM classifier.
- `gru`: unidirectional GRU classifier.
- `bigru`: bidirectional GRU classifier.
- `fcn`: 1D fully convolutional baseline for time-series classification.

## What is improved vs the original script

- split into multiple modules
- validation split instead of selecting on the test set
- best-checkpoint saving
- cosine learning-rate schedule
- gradient clipping
- deterministic data loading helpers
- stronger paper-backed baselines
- shift-stability metric on the test set
- front-end diagnostics for learned kernels
- comparison runner for multiple models and seeds

## Strong baseline rationale

The comparison suite now covers three complementary groups:

1. **Direct TCN baselines**: plain TCN and smoothed-TCN test whether gains come from anti-aliasing ideas or simply from more capacity.
2. **Fixed-filter TCN baselines**: Gaussian, Hamming, Savitzky-Golay, and moving average isolate the benefit of *learnability* versus classical smoothing.
3. **Cross-family baselines**: LSTM, GRU, BiLSTM, BiGRU, and FCN show whether the proposed method is competitive beyond the immediate TCN family.

## Quick start

```bash
pip install -r requirements.txt
```

Train plain TCN:

```bash
python train.py \
  --model tcn_plain \
  --output-dir ./outputs/tcn_plain
```

Train smoothed TCN:

```bash
python train.py \
  --model smoothed_tcn \
  --smoothed-tcn-smoother moving_avg \
  --smoothed-tcn-kernel-size 5 \
  --output-dir ./outputs/smoothed_tcn
```

Train FCN baseline:

```bash
python train.py \
  --model fcn \
  --fcn-channels 128,256,128 \
  --fcn-kernel-sizes 8,5,3 \
  --output-dir ./outputs/fcn
```

Train improved LPS-ConvPlus:

```bash
python train.py \
  --model lps_conv_plus \
  --causal \
  --front-residual \
  --dc-mode project \
  --kernel-init identity \
  --gate-init -4.0 \
  --output-dir ./outputs/lps_conv_plus
```

Run a broader comparison:

```bash
python compare_models.py \
  --models tcn_plain,smoothed_tcn,gaussian_tcn,savgol_tcn,lstm,gru,fcn,lps_conv_plus \
  --seeds 1111,2222,3333 \
  --output-dir ./outputs/compare
```

## Outputs

Each run directory contains:

- `history.csv`: epoch-by-epoch training/validation metrics
- `best.pt`: best checkpoint by validation accuracy
- `summary.json`: final metrics, config, and front-end diagnostics

The comparison runner also writes `comparison.csv`.

## Notes

- The default `lps_conv_plus` configuration starts near identity using `kernel-init identity` and `gate-init -4.0`, which is much safer than starting with an unconstrained random front-end.
- `dc-mode project` keeps the paper-style mean-matching behavior.
- `--normalize-kernel-dc` is available if you want exact kernel-sum normalization instead.
- The `smoothed_tcn` baseline is meant to be a strong degridding comparison inside the TCN stack, not just an input prefilter.
