import numpy as np

class NeuralNetwork:
    """
    A simple single-layer neural network with sigmoid activation.
    """
    def __init__(self, input_size=3, seed=1):
        """
        Initialize the neural network with random weights.
        :param input_size: Number of input features
        :param seed: Seed for random number generator
        """
        np.random.seed(seed)
        self.synaptic_weights = 2 * np.random.random((input_size, 1)) - 1

    def sigmoid(self, x):
        """
        Apply the sigmoid activation function.
        :param x: Input value or array
        :return: Sigmoid of the input
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Compute the derivative of the sigmoid function.
        :param x: Sigmoid output
        :return: Derivative of sigmoid
        """
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        """
        Train the neural network using the provided dataset.
        :param training_inputs: Input data for training
        :param training_outputs: Expected outputs for training
        :param training_iterations: Number of iterations for training
        """
        for iteration in range(training_iterations):
            # Forward pass
            output = self.think(training_inputs)
            
            # Compute error
            error = training_outputs - output
            
            # Compute adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            
            # Update weights
            self.synaptic_weights += adjustments

    def think(self, inputs):
        """
        Make a prediction based on input data.
        :param inputs: Input data
        :return: Output prediction
        """
        inputs = inputs.astype(float)
        return self.sigmoid(np.dot(inputs, self.synaptic_weights))


if __name__ == "__main__":
    # Initialize the neural network
    neural_network = NeuralNetwork()

    print("Random initial synaptic weights:")
    print(neural_network.synaptic_weights)

    # Training dataset
    training_inputs = np.array([[0, 0, 1],
                                 [1, 1, 1],
                                 [1, 0, 1],
                                 [0, 1, 1]])
    training_outputs = np.array([[0], [1], [1], [0]])

    # Train the neural network
    neural_network.train(training_inputs, training_outputs, 10000)

    print("\nSynaptic weights after training:")
    print(neural_network.synaptic_weights)

    # User input
    try:
        A = float(input("Input 1: "))
        B = float(input("Input 2: "))
        C = float(input("Input 3: "))
        
        user_input = np.array([A, B, C])
        print("\nNew situation: input data =", user_input)
        print("Output prediction:")
        print(neural_network.think(user_input))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
