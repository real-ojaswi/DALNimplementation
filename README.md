# DALN Implementation

This repository contains the full implementation of [Reusing the Task-Specific Classifier as a Discriminator: Discriminator-free Adversarial Domain Adaptation](https://openaccess.thecvf.com/content/CVPR2022/html/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.html) using TensorFlow.

## Overview

The implementation includes experiments conducted on the MNIST and USPS datasets, demonstrated in the following notebooks:
- `DALNCustom.ipynb`: Custom experiments and analysis.
- `DALNMNISTtoUSPS.ipynb`: Experiment specifically transferring from MNIST to USPS.

The model is trained on MNIST and evaluated on USPS dataset with and without Domain Adaptation techniques using DALN (Discriminator-free Adversarial Domain Adaptation).

## Components

- **DALNModel.py**: Implementation of the DALN model.
- **grl.py** and **nwd.py**: Implementations of the Gradient Reversal Layer and Nuclear Wasserstein Discrepancy, adapted from the [official implementation repository](https://github.com/xiaoachen98/DALN) originally written for PyTorch.
- **DALNtrain.py**: Utilities for training the DALN model.
- **DisplayLogs.py**: Utilities for logging and calculating accuracies during training.
- **DALNCustom.ipynb** and **DALNMNISTtoUSPS.ipynb**: Jupyter notebooks for conducting experiments and showcasing results.

## Results

The experimental results demonstrate a significant improvement in accuracy when using DALN compared to training without domain adaptation:

| Method       | Source Accuracy | Target Accuracy |
|--------------|-----------------|-----------------|
| Source only  | 0.9998          | 0.417           |
| DALN         | 0.9341          | 0.838           |

### Visualizations

#### TSNE Plots

- **Without DALN**: Features from MNIST and USPS datasets appear clearly separated.
  ![TSNE without DALN](https://github.com/thenoobcoderr/DALNimplementation/assets/139956609/acdaf4fc-2fd7-4479-b3f8-f1a4aad53783)

- **With DALN**: Features from both datasets show similar distributions, indicating successful domain adaptation.
  ![TSNE with DALN](https://github.com/thenoobcoderr/DALNimplementation/assets/139956609/d31ee3e1-92c7-48ba-8a47-b1e76751ca3a)

#### Determinacy and Diversity

DALN also enhances determinacy (confidence of predictions) and diversity (performance across different classes), as observed in the provided visualizations.

## Training Procedure

To train the model:

1. **Prepare Datasets**: Import and resize MNIST and USPS datasets to 32x32x3 to match model requirements.

2. **Initialize Model**: Import the model from `DALNModel.py` and initialize it.

3. **Training Setup**: Import `train` from `DALNtrain.py` and initialize a training object (`trainer`) with parameters such as `X_source`, `y_source`, `model`, `batch_size`, `X_target`, `y_target`, `epochs`, and `source_only` boolean.

4. **Run Training**: Set `source_only=True` for training without domain adaptation, or `source_only=False` for training with DALN.

5. **Predictions**: Use the `predict_label` method of the model object to predict labels.

## Calculating Accuracy

- During training, both source and target accuracies are displayed.
- Alternatively, use the `accuracy_score` function or import `display_logs` from `DisplayLogs.py` for automated accuracy calculation and logging.

## Originality of Code

- Codes in `grl.py` and `nwd.py` were adapted from the DALN repository originally written in PyTorch.
- The model architecture has been optimized for computational efficiency while maintaining effectiveness.

## Datasets

- **MNIST**: Handwritten digits dataset imported from `tensorflow.keras.datasets.mnist`.
- **USPS**: Handwritten digits dataset imported from `extra_keras.usps`.

Both datasets are small-sized with sufficient samples, making them suitable for domain adaptation experiments due to their notable differences.

## References

- [DALN Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Reusing_the_Task-Specific_Classifier_as_a_Discriminator_Discriminator-Free_Adversarial_Domain_CVPR_2022_paper.pdf)
- [DALN Repository](https://github.com/xiaoachen98/DALN)

Feel free to explore and contribute to this repository to enhance domain adaptation techniques using DALN.
