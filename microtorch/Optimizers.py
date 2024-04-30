

class SGD:
    def __init__(self, parameters, learning_rate=0.01):

        """
        Initialize the SGD optimizer.

        Args:
            parameters (callable): A function that returns the model parameters.
            learning_rate (float): The learning rate for the optimizer.
        """

        self.learning_rate = learning_rate
        self.parameters = parameters

    def zero_grad(self):

        """
        Clear the gradients of all model parameters.
        """

        for p in self.parameters():
            p.grad = 0.0

    def step(self):

        """
        Update model parameters using SGD.
        """
        
        for p in self.parameters():
            p.data += -self.learning_rate * p.grad
