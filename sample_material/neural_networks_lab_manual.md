# Deep Learning Lab Manual

## Experiment 1: Perceptron for Logic Gates

Aim: To implement a single-layer perceptron that learns the AND and OR gates.

Theory: A perceptron computes a weighted sum of its inputs plus a bias and applies a step activation function. The weights are updated using the perceptron learning rule: w = w + learning_rate × (target − output) × input. A single perceptron can only learn linearly separable functions, so it cannot learn the XOR gate.

Procedure:
1. Initialise the weights and bias to zero.
2. For every training pair compute the output using the step function.
3. Update the weights with the perceptron learning rule.
4. Repeat for several epochs until the error becomes zero.

Viva question: Why can a perceptron not learn XOR? Because the XOR outputs cannot be separated by a single straight line, i.e. XOR is not linearly separable. A multi-layer network is required.

## Experiment 2: Backpropagation in a Multi-Layer Network

Aim: To train a two-layer neural network with backpropagation on a small dataset.

Theory: Backpropagation computes the gradient of the loss with respect to every weight using the chain rule, moving from the output layer back to the input layer. Gradient descent then updates each weight: w = w − η ∂L/∂w, where η is the learning rate. The sigmoid activation can cause the vanishing gradient problem in deep networks, because its derivative is at most 0.25. The ReLU activation mitigates this problem.

Procedure:
1. Perform a forward pass and compute the loss.
2. Perform the backward pass to compute gradients with the chain rule.
3. Update the weights using gradient descent and repeat for all epochs.

## Experiment 3: Image Classification with a CNN on MNIST

Aim: To build a convolutional neural network that classifies handwritten digits.

Theory: A convolution layer slides small filters over the image to detect local features such as edges. A pooling layer reduces the spatial size of the feature maps. The MNIST dataset contains 60,000 training images and 10,000 test images of 28×28 grayscale digits. The output layer uses 10 units with softmax activation and the model is trained with categorical cross-entropy loss. Dropout with a rate of 0.5 is used to reduce overfitting.

Viva question: What is the purpose of pooling? It reduces the size of the feature maps, which lowers computation and makes the features more robust to small shifts.
