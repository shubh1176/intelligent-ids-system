# Network Signal Analysis Model

A deep learning model for network signal analysis using PyTorch, featuring residual learning, self-attention, and multi-scale processing.

## Architecture Overview

The model consists of three main components:

1. **Input Processing**
   - 1x1 Convolution for initial feature projection (input_dim → 128 channels)
   - Batch Normalization and Dropout regularization

2. **Feature Extraction**
   - Residual blocks with self-attention mechanism
   - Multi-scale processing (3,5,7 kernel sizes)
   - Feature fusion with channel calibration
   - Dropout layers for regularization

3. **Classification Head**
   - Adaptive Average Pooling
   - Three-layer classifier (256→128→64→num_classes)
   - Dropout regularization (0.3)

## Key Features

- **Self-Attention**: Captures long-range dependencies in network traffic
- **Residual Learning**: Enhanced gradient flow and deeper architecture
- **Multi-Scale Processing**: Captures patterns at different temporal scales
- **Dynamic Feature Fusion**: Adaptive feature combination
- **Strong Regularization**: Multiple dropout layers and batch normalization

## Training Configuration

- Optimizer: AdamW (lr=0.001, weight_decay=0.01)
- Loss: CrossEntropy with label smoothing (0.05)
- Learning Rate Schedule: OneCycleLR with warm restarts
- Early Stopping: Patience=5, Accuracy Threshold=99.99%
- Batch Size: 64

## Dataset

Uses the CICIDS2017 dataset for network intrusion detection.

For dataset details and citation, see original paper:


```bibtex
@inproceedings{sharafaldin2018toward,
    title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
    author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
    booktitle={4th International Conference on Information Systems Security and Privacy (ICISSP)},
    year={2018},
    address={Portugal}
}
