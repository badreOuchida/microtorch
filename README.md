# Neural Network Gradient Computation Project

## Overview

This project is inspired by PyTorch and aims to implement a framework similar to PyTorch. Its purpose is to provide a hands-on understanding of the underlying mechanisms of PyTorch by building similar functionalities from scratch. Through this project, developers can gain insights into the inner workings of PyTorch, including concepts such as automatic differentiation, neural network layers, optimization algorithms, and loss functions.

This project implements a neural network framework for computing gradients using backpropagation. It includes custom implementations of key components such as the Value class for automatic differentiation, SGD optimizer, MSE loss function, MSE with L2 regularization loss function, Neuron class, and Layer class.

This is a basic implementation, but my goal is to gradually add functionality until implementing all the necessary components to build fully optimized neural network architectures, including CNNs, RNNs, and extending to transformers.

## Components

1. **Value Class**: The Value class serves as the core component for automatic differentiation. It represents a value in a computational graph and supports various operations for calculating gradients.

2. **SGD Optimizer**: The SGD (Stochastic Gradient Descent) optimizer is implemented to update model parameters using gradient descent.

3. **Loss Functions**:

   - **Mean Squared Error (MSE) Loss Function**: Calculates the mean squared error between predicted and true values.
   - **MSE with L2 Regularization Loss Function**: Extends MSE by adding L2 regularization to prevent overfitting.

4. **Neuron Class**: Represents a single neuron in a neural network, with support for various activation functions.

5. **Layer Class**: Represents a layer of neurons in a neural network.

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/badreOuchida/microtorch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Example Usage

You can find an example of usage in `demo.ipynb`, where there is a demonstration of how to utilize microtorch.

### Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, feel free to open an issue or submit a pull request.
