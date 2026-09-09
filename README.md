# Approximate Machine Unlearning with Gradient Ascent

## Overview

This project explores the problem of machine unlearning: reducing the influence of selected training data from a trained machine learning model without completely retraining the model from scratch.

The project compares three models:

1. **Full Model**: trained using the complete dataset.
2. **Unlearned Model**: starts from the full model and applies a first-order gradient-ascent update on the selected forgotten data.
3. **Retrained Model**: trained from scratch after removing the forgotten data.

The retrained model provides a reference for evaluating how closely the approximate unlearning procedure approaches retraining.

## Method

The project uses a small multilayer perceptron (MLP) trained on a subset of the MNIST dataset.

The final experiment uses:

- Dataset: MNIST
- Experimental training subset: 5,000 samples
- Forget set: 100 samples
- Retain set: 4,900 samples
- Model: MLP with 25,450 parameters
- Optimizer: Adam
- Training epochs: 5
- Unlearning method: First-order gradient ascent
- Gradient-ascent step size: 0.1

The project also explores the theoretical basis of second-order unlearning using the Hessian and Hessian-vector products (HVPs).

## Hessian-Based Formulation

At a trained optimum:

\[
\nabla L_{all}(\theta^*) \approx 0
\]

Since:

\[
L_{all} = L_R + L_F
\]

we obtain:

\[
\nabla L_R(\theta^*) \approx -\nabla L_F(\theta^*)
\]

Using a second-order approximation gives:

\[
H_R \Delta\theta \approx \nabla L_F
\]

and therefore:

\[
\Delta\theta \approx H_R^{-1}\nabla L_F
\]

Instead of explicitly constructing the full Hessian, the project demonstrates Hessian-vector products.

A full iterative second-order solve was found to be computationally impractical on the available CPU, so the final practical experiment uses a first-order baseline.

## Results

| Model | Forget Accuracy | Retain Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Full Model | 91.00% | 91.06% | 89.07% |
| Unlearned Model | 87.00% | 90.45% | 88.10% |
| Retrained Model | 90.00% | 92.78% | 89.88% |

### Unlearning Effect

- Forgotten-data accuracy decreased by **4.00 percentage points**.
- Retained-data accuracy decreased by only **0.61 percentage points**.
- Forgotten-data loss increased from **0.3639 to 0.4429** after the first-order update.

### Parameter Distance

- Full vs Unlearned relative distance: **0.0091**
- Unlearned vs Retrained relative distance: **1.2614**

## Conclusion

The experiment demonstrates **partial approximate machine unlearning** using a first-order gradient-ascent update.

The forgotten subset became less well represented by the model while performance on the retained data changed only slightly. However, the unlearned model remained substantially different from the retrained model in parameter space.

Therefore, the experiment should not be interpreted as exact data erasure. It provides a practical baseline and demonstrates the theoretical motivation for more sophisticated second-order machine unlearning methods.

## Technologies

- Python
- PyTorch
- Torchvision
- MNIST
- Google Colab

## Project Structure

```text
machine-unlearning/
├── machine_unlearning.ipynb
└── README.md
