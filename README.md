# Network Signal Analysis Model

A deep learning model for network signal analysis using PyTorch, featuring dynamic feature fusion and adaptive channel calibration.

## Architecture Overview

The model consists of three main components:

1. **Input Projection Layer**
   - 1x1 Convolution for initial feature extraction
   - Batch Normalization and ReLU activation
   - Output: 64 channels

2. **Dual Processing Blocks**
   Each block contains:
   - Dynamic Feature Fusion (DFF)
     - Multi-scale feature extraction (3,5 window sizes)
     - Adaptive feature weighting
     - 1x1 Convolution for dimension reduction
   - Adaptive Channel Calibration (ACC)
     - Parallel avg/max pooling
     - MLP-based channel attention
   - Skip connections with ReLU activation

3. **Classification Head**
   - Adaptive Average Pooling
   - Two-layer classifier (64→32→num_classes)
   - Dropout regularization (0.3)

## Key Features

- **Dynamic Feature Processing**: Multi-scale feature extraction with adaptive fusion
- **Channel Attention**: Learnable channel-wise feature calibration
- **Residual Connections**: Enhanced gradient flow with skip connections
- **Label Smoothing**: Improved generalization with smoothed targets
- **Cosine Learning Rate**: Warm restarts for optimal convergence

## Training Configuration

- Optimizer: AdamW (lr=0.0005, weight_decay=0.01)
- Loss: CrossEntropy with label smoothing (0.05)
- Learning Rate Schedule: CosineAnnealingWarmRestarts
- Early Stopping: Patience=10, Accuracy Threshold=99.5%
- Batch Size: 64
- Sequence Processing: 100-sample windows, 50-sample stride

## Performance Monitoring

- Real-time loss and accuracy tracking
- Training history visualization
- Early stopping based on validation metrics

---

## Dataset

This project uses the [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html), a widely recognized dataset for network intrusion detection research. 

- **Usage**:
  - The dataset must be downloaded and appropriately preprocessed before use in the IDS.

### Citation


```bibtex
@inproceedings{sharafaldin2018toward,
    title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
    author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
    booktitle={4th International Conference on Information Systems Security and Privacy (ICISSP)},
    year={2018},
    address={Portugal}
}
