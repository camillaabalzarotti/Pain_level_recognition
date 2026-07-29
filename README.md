# Multivariate Time-Series Classification for Pain-Level Recognition

This project investigates deep-learning approaches for classifying pain levels from multivariate sequential data. The objective is to predict one of three classes—**no pain**, **low pain**, or **high pain**—from joint-motion measurements and pain-related survey features.

The complete methodology, experiments, and results are available in the [final report](Report.pdf).

Each notebook represents different tries that were experimented. The "Models" folder contains the three best L-SFAN models that are used
at the end of the Transformer_and_L-SFAN.ipynb to build the final ensemble. As indicated with a #TODO comment, please change the path in the notebook accordingly to your models' location.

---

## Project Overview

The project compares temporal and spatial deep-learning architectures for automatic pain recognition:

- **Bidirectional GRU**, focused on short-range temporal dynamics
- **Transformer**, designed to capture local and long-range dependencies
- **L-SFAN**, a lightweight spatial-attention model for coordinated joint activity
- **Ensembles** of specialised L-SFAN models
- **Manual and GAN-based data augmentation** to reduce class imbalance

The main finding is that model architecture and sequence preprocessing are strongly interconnected. Spatial attention performed particularly well, while a windowed BiGRU achieved the best Kaggle score by learning local temporal patterns from overlapping subsequences.

---

## Dataset

The dataset contains recordings from **661 users**, referred to as pirates in the challenge.

Each user contributes:

- 160 temporal samples
- 31 joint-angle features
- 4 pain-survey features
- 1 target pain class

### Target Classes

- No pain
- Low pain
- High pain

### Main Challenges

- Imbalanced class distribution
- Only one complete sequence per user
- High-dimensional input space
- Risk of overfitting and user-specific correlations
- Limited variability in the available training data

---

## Data Preparation

The preprocessing pipeline includes:

- Conversion of categorical attributes to numerical values
- Min–Max feature normalisation
- User-based training and validation splitting
- Weighted loss functions for class imbalance
- Full-sequence inputs for Transformer and L-SFAN models
- Windowing and striding for the GRU model

The user-based split ensures that sequences from the same user do not appear in both training and validation sets, reducing information leakage and providing a more realistic estimate of generalisation to unseen users.

---

## Models

### Bidirectional GRU

The GRU models sequential dependencies using forward and backward recurrent passes.

The best configuration uses:

- 2 recurrent layers
- Hidden size of 128
- Dropout of 0.1
- Weighted cross-entropy loss
- Reduce-on-Plateau learning-rate scheduling

The original sequences are divided into short, partially overlapping windows. This increases the number of training examples and helps the model capture fine-grained local temporal dynamics.

---

### PiratePainTransformer

The Transformer uses self-attention to model both local and long-range temporal relationships.

Main components:

- 4 Transformer encoder layers
- Model dimension of 128
- 8 attention heads
- GELU activations
- Residual connections
- Attention pooling over time

The attention-pooling layer learns to emphasise the time steps that are most informative for pain classification.

---

### PiratePain L-SFAN

The Lightweight Spatially-Focused Attention Network is adapted to multivariate motion data.

The architecture includes:

- Two convolutional blocks
- Batch normalisation and ReLU activation
- Temporal average pooling
- Multi-head spatial attention
- Five attention heads

L-SFAN focuses on coordinated activity across joints and pain-related features, prioritising spatial relationships over temporal order.

A five-fold cross-validation experiment was also performed to assess stability across different user splits.

---

## Data Augmentation

Two complementary augmentation strategies were explored.

### Manual Augmentation

Manual augmentation was applied mainly to minority classes and included:

- Small Gaussian jitter on dynamic features
- Limited temporal shifts
- Edge padding to preserve motion continuity

These transformations introduce controlled variability without creating unrealistic discontinuities.

### Conditional WGAN

A conditional Wasserstein GAN was trained to generate synthetic multivariate sequences conditioned on the target pain class.

The generator produces pain-class-specific sequences, while the critic encourages realistic temporal behaviour. GAN training required careful tuning to reduce the risk of mode collapse.

---

## Results

| Model | Kaggle F1 | Validation Accuracy | Validation F1 |
|---|---:|---:|---:|
| BiGRU baseline | 0.8885 | 0.8788 | 0.7230 |
| Transformer | 0.8212 | 0.9092 | 0.8307 |
| L-SFAN | 0.9135 | **0.9596** | 0.9241 |
| GAN L-SFAN | 0.9144 | 0.9394 | 0.8897 |
| GAN Transformer | 0.9146 | 0.9192 | 0.8562 |
| Manually augmented L-SFAN | 0.9156 | 0.9293 | 0.8891 |
| L-SFAN ensemble | 0.9244 | — | — |
| Five-fold CV L-SFAN | 0.9091 | 0.9323 | 0.8354 |
| BiGRU revisited | **0.9365** | 0.9269 | **0.9261** |

### Key Results

- The standard L-SFAN achieved the highest validation accuracy: **95.96%**.
- The revisited BiGRU achieved the highest Kaggle F1 score: **0.9365**.
- The revisited BiGRU also achieved the highest reported validation F1 score: **0.9261**.
- The L-SFAN ensemble reached a Kaggle F1 score of **0.9244**.
- Transformer models achieved strong validation accuracy but generally underperformed L-SFAN and the windowed BiGRU.

---

## Main Findings

- Spatial relationships between joint features are highly informative for pain recognition.
- L-SFAN effectively captures coordinated joint activations through spatial attention.
- GRU windowing and striding improve generalisation by exposing the model to diverse local temporal contexts.
- Full-sequence Transformers may require larger datasets to learn robust attention patterns.
- Manual and GAN-based augmentation help reduce class imbalance.
- Different augmentation strategies can specialise models toward different pain classes.
- Combining specialised models in an ensemble improves robustness, although it reduces interpretability and increases computational cost.

---

## Limitations

- The dataset is small and class-imbalanced.
- Each user contributes only one complete sequence.
- GAN-generated sequences do not always reproduce the full variability of genuine pain expressions.
- Conditional WGAN training is sensitive to mode collapse.
- Training multiple models and ensembles increases computational cost.
- Ensemble predictions are more difficult to interpret than predictions from a single model.

---

## Future Work

Possible future improvements include:

- Time-series-specific augmentation using TimeGAN
- Better preservation of temporal dynamics in synthetic samples
- Visualisation of the most influential temporal segments
- Improved explainability for spatial and temporal attention
- Evaluation on larger and more diverse clinical datasets

---

## Authors

- Camilla Balzarotti
- Emma Battaglia
- Mathieu Alexandre Lavoie

---

## Disclaimer

This repository contains an academic machine-learning project. It is not a certified medical system and must not be used for clinical diagnosis or treatment decisions.
