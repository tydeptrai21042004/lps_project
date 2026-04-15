from __future__ import annotations

PAPER_BASELINE_MODELS = (
    "tcn_plain",
    "hybrid_dilated_tcn",
    "smoothed_tcn",
    "blurpool_tcn",
    "bilstm",
    "bigru",
    "fcn",
)

PROPOSAL_MODELS = (
    "lps_conv",
)

ABLATION_MODELS = (
    "learnable_front_tcn",
    "gaussian_tcn",
    "hamming_tcn",
    "savgol_tcn",
    "moving_avg_tcn",
    "tcn_bn",
    "tcn_attn",
    "tcn_strong",
    "lstm",
    "gru",
)

MODEL_PAPER_SUPPORT = {
    "tcn_plain": {
        "kind": "paper_baseline",
        "paper": "Bai, Kolter, Koltun (2018), An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling",
        "note": "standard dilated TCN backbone baseline",
    },
    "hybrid_dilated_tcn": {
        "kind": "paper_baseline",
        "paper": "Wang et al. (HDC, 2018), Understanding Convolution for Semantic Segmentation",
        "note": "HDC-inspired non-gridding dilation schedule baseline for 1-D TCN",
    },
    "smoothed_tcn": {
        "kind": "paper_baseline",
        "paper": "Wang and Ji (2018), Smoothed Dilated Convolutions for Improved Dense Prediction",
        "note": "smoothed dilated-convolution baseline",
    },
    "blurpool_tcn": {
        "kind": "paper_baseline",
        "paper": "Zhang (2019), Making Convolutional Networks Shift-Invariant Again",
        "note": "fixed anti-aliased front-end baseline using blurpool-style low-pass filtering",
    },
    "bilstm": {
        "kind": "paper_baseline",
        "paper": "Schuster and Paliwal (1997), Bidirectional Recurrent Neural Networks; Hochreiter and Schmidhuber (1997), Long Short-Term Memory",
        "note": "standard bidirectional recurrent baseline",
    },
    "bigru": {
        "kind": "paper_baseline",
        "paper": "Schuster and Paliwal (1997), Bidirectional Recurrent Neural Networks; Cho et al. (2014), RNN Encoder-Decoder / GRU",
        "note": "standard bidirectional recurrent baseline",
    },
    "fcn": {
        "kind": "paper_baseline",
        "paper": "Wang, Yan, Oates (2016/2017), Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline",
        "note": "strong FCN time-series baseline",
    },
    "lps_conv": {
        "kind": "proposal",
        "paper": "This paper",
        "note": "single-stage symmetric linear-phase front end",
    },
    "learnable_front_tcn": {
        "kind": "ablation",
        "paper": "none (internal ablation)",
        "note": "unconstrained learnable front-end ablation",
    },
    "gaussian_tcn": {
        "kind": "ablation",
        "paper": "classical DSP low-pass smoother",
        "note": "fixed Gaussian front-end ablation",
    },
    "hamming_tcn": {
        "kind": "ablation",
        "paper": "classical DSP windowed smoother",
        "note": "fixed Hamming front-end ablation",
    },
    "savgol_tcn": {
        "kind": "ablation",
        "paper": "Savitzky-Golay classical smoothing filter",
        "note": "fixed Savitzky-Golay front-end ablation",
    },
    "moving_avg_tcn": {
        "kind": "ablation",
        "paper": "classical moving-average smoother",
        "note": "fixed moving-average front-end ablation",
    },
    "tcn_bn": {
        "kind": "ablation",
        "paper": "none (internal optimization variant)",
        "note": "batch-normalized TCN optimization ablation",
    },
    "tcn_attn": {
        "kind": "ablation",
        "paper": "none (internal optimization variant)",
        "note": "attention-pooling TCN optimization ablation",
    },
    "tcn_strong": {
        "kind": "ablation",
        "paper": "none (internal optimization variant)",
        "note": "strong tuned TCN optimization ablation",
    },
    "lstm": {
        "kind": "ablation",
        "paper": "Hochreiter and Schmidhuber (1997), Long Short-Term Memory",
        "note": "unidirectional recurrent ablation",
    },
    "gru": {
        "kind": "ablation",
        "paper": "Cho et al. (2014), RNN Encoder-Decoder / GRU",
        "note": "unidirectional recurrent ablation",
    },
}

ALL_MODELS = tuple(MODEL_PAPER_SUPPORT.keys())
