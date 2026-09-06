# Approximate Machine Unlearning with Gradient and Hessian-Based Methods

## Overview

Machine unlearning aims to reduce the influence of selected training data from an already-trained machine learning model without completely retraining the model from scratch.

This project explores a gradient- and Hessian-based approach to approximate machine unlearning using PyTorch and the MNIST dataset.

## Objective

The experiment compares three models:

1. **Full Model** - trained using the complete training subset.
2. **Unlearned Model** - starts from the full model and applies a first-order gradient-ascent update using the data selected for forgetting.
3. **Retrained Model** - trained from scratch after removing the forgotten data, serving as a reference.

## Method

The theoretical formulation is based on the relationship between the retained-data loss and the forgotten-data gradient.

A second-order approximation gives:

`Δθ ≈ H⁻¹∇L_F`

where:

- `∇L_F` is the gradient of the forgotten-data loss.
- `H` is the Hessian of the retained-data loss.
- `Δθ` is the approximate parameter update.

Instead of explicitly constructing the Hessian, the project demonstrates the use of **Hessian-vector products (HVPs)**.

A full iterative second-order solve was found to be computationally impractical on the available CPU, so the final experiment uses a first-order gradient-ascent baseline.

## Experimental Setup

- Dataset: MNIST
- Experimental subset: 5,000 training examples
- Forget set: 100 examples
- Retain set: 4,900 examples
- Model: Small MLP
- Parameters: 25,450
- Optimizer: Adam
- Training epochs: 5
- Framework: PyTorch

## Results

| Model | Forget Accuracy | Retain Accuracy | Test Accuracy |
|---|---:|---:|---:|
| Full | 91.00% | 91.06% | 89.07% |
| Unlearned | 87.00% | 90.45% | 88.10% |
| Retrained | 90.00% | 92.78% | 89.88% |

### Observations

- Forgotten-data accuracy decreased by **4.00 percentage points** after the unlearning update.
- Retained-data accuracy decreased by only **0.61 percentage points**.
- The unlearned model was not identical to the retrained model in parameter space.

## Conclusion

The experiment demonstrates **partial approximate machine unlearning**, rather than exact data erasure.

The results show that a first-order update can reduce the model's performance on selected forgotten data while largely preserving performance on retained data.

The project also demonstrates the theoretical and computational use of Hessian-vector products as a foundation for more sophisticated second-order unlearning methods.

## Limitations

- The experiment uses a relatively small MNIST subset.
- The final practical method is first-order rather than a complete Hessian-inverse solution.
- A larger-scale experiment would be required to evaluate scalability and stronger unlearning guarantees.

## Technologies

- Python
- PyTorch
- Torchvision
- Google Colab
- MNIST
