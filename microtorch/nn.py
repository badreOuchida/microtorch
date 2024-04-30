import numpy as np

from microtorch.utils import generator

class Model :             
    def __call__(self,xs) :
        xs = np.array(xs) if isinstance(xs, list) else xs

        if isinstance(xs, np.ndarray) and xs.ndim == 1 :
            xs = xs.reshape(1,-1)
        outs = [ self.forward(x) for x in xs ]
        return [ out[0] for out in outs ]

    def parameters(self) : 
        return np.concatenate([n.parameters() for n in self.layers])



class Neuron :

    """
    A class representing a single neuron in a neural network with support for various activation functions.

    Attributes:
        w (numpy.ndarray): Weights of the neuron.
        b (numpy.ndarray): Bias of the neuron.
        activation (str): The activation function to use ('tanh', 'relu', or 'sigmoid').
    """

    def __init__(self, nin , activation = 'tanh' ):

        """
        Initialize a Neuron object.

        Args:
            nin (int): The number of input features.
            activation (str): The activation function to use. Default is 'tanh'.
        """

        # Initialize weights and bias using the generator function from the Value module
        self.w = generator(nin)  # Generate weights     
        self.b = generator(1)  # Generate bias
        self.activation = activation # Set activation function
    

    def _activation(self, z):
        """
        Apply the activation function to the linear transformation output.

        Args:
            z (numpy.ndarray): Linear transformation output.

        Returns:
            Value: Output of the neuron after applying the activation function.
        """
        if self.activation == 'tanh':
            return z[0].tanh()  # Tanh activation
        elif self.activation == 'relu':
            return z[0].relu()  # ReLU activation
        elif self.activation == 'sigmoid':
            return z[0].sigmoid()  # Sigmoid activation
        else:
            return z[0]
            
        

    def __call__(self,x) :

        """
        Compute the output of the neuron given input data.

        Args:
            x (numpy.ndarray or list): Input data.

        Returns:
            numpy.ndarray: Output of the neuron after applying the activation function.
        """

        # Convert input to numpy array if it's a list
        _x = np.array(x) if isinstance(x, list) else x

        # Reshape input if it's a 1D array
        if isinstance(_x, np.ndarray) and _x.ndim == 1 :
            _x = _x.reshape(1,-1)
        
        # Compute the linear transformation
        z = np.dot(_x,self.w) + self.b


        # Apply the activation function
        

        return self._activation(z)
   
    def parameters(self) :

        """
        Get the parameters (weights and bias) of the neuron.

        Returns:
            numpy.ndarray: Concatenated array of weights and bias.
        """

        # Return the parameters (weights and bias)
        return np.concatenate((self.w,self.b))

class Layer :

    """
    A class representing a layer of neurons in a neural network.

    Attributes:
        layer (numpy.ndarray): Array of neurons in the layer.
    """

    def __init__(self,nin , nout , activation = 'tanh'):

        """
        Initialize a Layer object.

        Args:
            nin (int): Number of input features.
            nout (int): Number of neurons in the layer.
            activation (str): Activation function to use. Default is 'tanh'.
        """

        # Create an array of neurons with specified input features and activation function
        self.layer = np.array([ Neuron(nin,activation=activation) for _ in range(nout) ])
        

    def __call__(self , x):

        """
        Compute the output of the layer given input data.

        Args:
            xs (numpy.ndarray or list): Input data.

        Returns:
            numpy.ndarray: Output of the layer after applying the activation function.
        """

        # Apply each neuron in the layer to the input data
        forward = np.vectorize(lambda n : n(x))
        outs = forward(self.layer)
        return outs
    
    def parameters(self):

        """
        Get parameters of the layer.

        Returns:
            numpy.ndarray: Concatenation of parameters of all neurons in the layer.
        """

        return np.concatenate([n.parameters() for n in self.layer])





