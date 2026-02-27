# mixTCC: Mixed-View Time-Series Contrastive Learning

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**mixTCC** is a framework for Time-Series Representation Learning. It leverages different contrastive learning views (Self-View, Cross-View, and Mixed-View) to extract robust features from 1D time-series data. This repository contains the code for pre-training, fine-tuning, and evaluating the models on various tasks, including mechanical fault diagnosis (CWRU, pFD) and physiological signal classification (Epilepsy).

## Repository Structure

```text
mixTCC/
├── dataloader/                         # Dataloaders for different datasets
├── models/                             # Network architectures (Encoders, Projectors, Classifiers)
├── trainer/                            # Training and evaluation loops (Pre-train & Fine-tune)
├── main.py                             # Main entry point for running experiments
├── preprocess_epilepsy.py              # Data preprocessing for the Epilepsy dataset
├── process_cwru_4class.py              # Data preprocessing for CWRU 4-Class task
├── process_cwru_ball_severity.py       # Data preprocessing for CWRU Ball Severity task
├── process_pFD_classify_5120-2560.py   # Data preprocessing & downsampling (5120->2560) for pFD 
├── utils.py                            # Utility functions (logging, metrics, etc.)
└── README.md                           # Project documentation
```

## Supported Datasets & Tasks
The repository evaluates the framework across different domains:

CWRU (Case Western Reserve University): Bearing fault diagnosis.

Tasks: 4-Class Classification & Fault Severity Classification (e.g., Ball Severity).

pFD (Paderborn University): Motor condition monitoring.

Tasks: Within-domain classification (pFD_a, pFD_b, etc.) and Cross-domain transfer (a2b, b2a, c2d, etc.).

Includes custom resampling scripts (e.g., 64kHz to 32kHz / 5120 to 2560 length).

Epilepsy: Epileptic Seizure Recognition dataset (EEG signals).

## Getting Started
1. Data Preprocessing
Before training, run the specific preprocessing script for your target dataset to format the data and generate the .pt files.

```text
# For Epilepsy dataset
python preprocess_epilepsy.py

# For pFD dataset (includes downsampling 5120 -> 2560)
python process_pFD_classify_5120-2560.py

# For CWRU 4-Class dataset
python process_cwru_4class.py

# For CWRU Ball Severity dataset
python process_cwru_ball_severity.py
```
2. Training and Evaluation
The main.py script acts as the entry point for both pre-training the representations and fine-tuning them on downstream tasks.

(Note: Adjust the arguments below based on your specific main.py implementation)

```text
# Example command to run the pipeline
python main.py --dataset cwru --view mixed --labeled_ratio 0.1
```

## Experiments & Evaluation Protocol
The framework systematically compares different learning strategies:

Baselines: Fully Supervised

Contrastive Views: Self-View, Cross-View, and Mixed-View (Ours)

Evaluation Metrics:

Semi-supervised / Few-shot evaluation: Fine-tuning the pre-trained encoder using limited labeled data fractions (1%, 5%, 10%, 20%, 50%).

Transfer Learning: Evaluating domain adaptation performance (e.g., training on pFD_a and testing on pFD_b).

Performance Metric: Macro F1-Score & Accuracy.

📄 License
This project is licensed under the terms of the LICENSE included in the repository.
