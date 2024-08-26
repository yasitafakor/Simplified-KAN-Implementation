# Simplified KAN Implementation

## Kolmogorov-Arnold Networks (KANs)

## Overview

This repository contains an implementation of **Kolmogorov-Arnold Networks (KANs)**, a neural network architecture inspired by the Kolmogorov-Arnold representation theorem. This theorem states that any multivariate continuous function can be decomposed into a finite number of continuous functions of a single variable, enabling a powerful way to model complex relationships.

KANs leverage this theorem by utilizing spline-based layers that allow for adaptive and efficient function approximation. Each layer in a KAN is designed to map inputs to outputs using a set of coefficients and grids that are adjusted during training. This makes KANs particularly suitable for tasks where interpretability and flexibility are crucial.

## Code Overview

### 1. `KANLayer` Class

The `KANLayer` class defines a single layer of a KAN, which uses splines for function approximation. Key components include:

- **Grid and Coefficient Initialization**: The grid points and coefficients are initialized based on the input and output dimensions. Noise is added to the grid during initialization to improve robustness.
- **Forward Pass**: The layer processes the input by computing the spline-based output and applying scaling factors.
- **Grid Update Mechanism**: The grid points can be updated based on new samples, allowing the layer to adapt dynamically.
- **Grid Initialization from Parent Layer**: A mechanism to initialize the grid from a parent model, facilitating transfer learning or hierarchical model design.

### 2. Spline Functions

- **`coef2curve`**: This function evaluates a spline curve at given input points (`x_eval`) using the provided grid and coefficients.
- **`curve2coef`**: This function calculates the coefficients for a spline that fits the given data points (`x_eval`, `y_eval`) based on the grid.
- **`B_batch`**: This function generates the B-spline basis matrix for a batch of inputs, which is crucial for the spline calculations.



### 3. `KAN` Class

The `KAN` class defines the overall network, which consists of multiple `KANLayer` instances stacked together. The key functionalities include:

- **Layer Initialization**: The network is initialized based on the dimensions specified in `layer_dims`. Each layer is an instance of `KANLayer`.
- **Grid Update and Initialization**: Functions to update the grid based on new samples (`update_grids`) and initialize grids from a parent model (`initialize_grids`).
- **Forward Pass**: The `forward` method processes the input through each layer, capturing pre-activations, post-activations, and post-spline activations for further analysis.

### 2. `KANLayer` Class

Each `KANLayer` is responsible for mapping inputs to outputs using adaptive spline functions. Key features include:

- **Grid and Coefficient Initialization**: Grids and coefficients are initialized with optional noise to improve training robustness.
- **Forward Pass**: Computes the spline-based output and applies scaling factors.
- **Grid Update Mechanism**: Grids can be dynamically updated based on new input samples, ensuring the model adapts to changing data distributions.

### 3. Example Usage

The code demonstrates how to use the `KAN` model to classify the Iris dataset. This includes:

- **Data Preparation**: Loading and preprocessing the Iris dataset.
- **Model Initialization**: Creating a `KAN` model with specified layer dimensions.
- **Training Loop**: Training the model with the Iris dataset, logging training losses and test accuracies.
- **Evaluation and Visualization**: Plotting the training loss and test accuracy to assess the model's performance.

### Visualization
![KAN_result](https://github.com/user-attachments/assets/5e7dcc32-74c5-4e8c-bc22-3404b07e7391)


## Original Paper

For more detailed information on the theoretical background and original implementation of KAN, please refer to the original paper [here](<https://arxiv.org/abs/2404.19756>).

