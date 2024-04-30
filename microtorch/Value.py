import math

class Value :

    """
    A class representing a value in a computational graph with support for automatic differentiation.
    """


    def __init__(self, data, _op='', _prev=(), label=''):

        """
        Initialize a Value object.

        Args:
            data: The data value.
            _op: The operation performed on this value (optional).
            _prev: A tuple of previous Value objects that influenced this value (optional), used to keep track of the computational graph.
            label: A label for this value (optional).
        """

        if isinstance(data, Value):
            # If data is a Value object, copy its attributes
            self.data = data.data
            self._op = data._op
            self._prev = data._prev
            self.label = data.label
            self.grad = data.grad
            self._backward = data._backward
        else:
            # Otherwise, initialize attributes
            self.data = data
            self._prev = set(_prev)
            self._op = _op
            self.label = label

            # Initialize gradient to zero
            self.grad = 0.0
            # Initialize backward function to a default empty lambda function
            self._backward = lambda: None


    def __repr__(self):

        """
        Return a string representation of the Value object.
        """

        return f"Value(data={self.data})"
    


    def __add__(self, other):

        """
        Perform addition operation between two Value objects.

        Args:
            other: The other Value object to add.

        Returns:
            A new Value object representing the result of the addition.
        """

        # Convert 'other' to a Value instance if it's not already
        other = other if isinstance(other, Value) else Value(other)

        # Create a new Value object representing the result of the addition
        out = Value(self.data + other.data, _op='+', _prev=(self, other))

        # Define the backward function for computing gradients
        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out

        
    def __radd__(self,other):

        """
        Perform addition operation with the Value object on the right side.

        Args:
            other: The other object to add.

        Returns:
            A new Value object representing the result of the addition.
        """

        # Delegate addition to __add__ method and return the result
        return self + other

    def __neg__(self):

        """
        Negate the Value object.

        Returns:
            A new Value object representing the negation of the original value.
        """

        # Multiply the Value object by -1 to perform negation
        return self * -1


    def __sub__(self, other):

        """
        Perform subtraction operation between two Value objects.

        Args:
            other: The other Value object to subtract.

        Returns:
            A new Value object representing the result of the subtraction.
        """

        # Delegate subtraction to __add__ method with negated 'other' and return the result
        return self + (-other)
        


    def __rsub__(self, other):

        """
        Perform subtraction operation with the Value object on the right side.

        Args:
            other: The other object to subtract.

        Returns:
            A new Value object representing the result of the subtraction.
        """

        # Delegate subtraction to __sub__ method and return the result
        return self - other

    
    def __mul__(self, other):

        """
        Perform multiplication operation between two Value objects.

        Args:
            other: The other Value object to multiply.

        Returns:
            A new Value object representing the result of the multiplication.
        """

        # Convert 'other' to a Value instance if it's not already
        other = other if isinstance(other, Value) else Value(other)
        
        # Calculate the product of the two values and create a new Value object
        out = Value(self.data * other.data, _op='*', _prev=(self, other))

        # Define the backward function for computing gradients
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out


    def __rmul__(self, other):

        """
        Perform multiplication operation with the Value object on the right side.

        Args:
            other: The other object to multiply.

        Returns:
            A new Value object representing the result of the multiplication.
        """

        # Delegate multiplication to __mul__ method and return the result
        return self * other

    
    
    def __pow__(self, pow):

        """
        Raise the Value object to the power of 'pow'.

        Args:
            pow: The exponent to raise the value to.

        Returns:
            A new Value object representing the result of the exponentiation.
        """

        assert isinstance(pow, (int, float)), "only supporting int/float powers for now"

        # Calculate the result of raising the value to the power of 'pow' and create a new Value object
        out = Value(self.data ** pow, _op=f'**{pow}', _prev=(self, ))

        # Define the backward function for computing gradients
        def _backward():
            self.grad += pow * (self.data ** (pow - 1)) * out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out


    def __rpow__(self, other):

        """
        Raise the other object to the power of the Value object.

        Args:
            other: The base number.

        Returns:
            A new Value object representing the result of the exponentiation.
        """

        # Calculate the result of raising 'other' to the power of the value and return it
        return other ** self.data


    def __truediv__(self, other):

        """
        Perform true division operation between two Value objects.

        Args:
            other: The divisor Value object.

        Returns:
            A new Value object representing the result of the division.
        """

        # Check if the divisor is zero
        if other == 0 or (isinstance(other, Value) and other.data == 0):
            raise ValueError("Division by zero is not allowed.")
        
        # Compute the result of the division and return it
        return self * (other ** -1)


    def __rtruediv__(self, other):

        """
        Perform true division operation with the Value object on the right side.

        Args:
            other: The other object to divide.

        Returns:
            A new Value object representing the result of the division.
        """

        # Delegate division to __truediv__ method and return the result
        return other / self


    # order

    def __lt__(self, other):

        """
        Compare if the Value object is less than the other object.

        Args:
            other: The other object to compare against.

        Returns:
            True if the Value object is less than the other object, False otherwise.
        """

        # Convert 'other' to a Value instance if it's not already and perform the comparison
        other = other if isinstance(other, Value) else Value(other)

        return self.data < other.data


    def __le__(self, other):

        """
        Compare if the Value object is less than or equal to the other object.

        Args:
            other: The other object to compare against.

        Returns:
            True if the Value object is less than or equal to the other object, False otherwise.
        """

        # Convert 'other' to a Value instance if it's not already and perform the comparison
        other = other if isinstance(other, Value) else Value(other)

        return self.data <= other.data


    def __gt__(self, other):

        """
        Compare if the Value object is greater than the other object.

        Args:
            other: The other object to compare against.

        Returns:
            True if the Value object is greater than the other object, False otherwise.
        """

        # Convert 'other' to a Value instance if it's not already and perform the comparison
        other = other if isinstance(other, Value) else Value(other)

        return self.data > other.data


    def __ge__(self, other):

        """
        Compare if the Value object is greater than or equal to the other object.

        Args:
            other: The other object to compare against.

        Returns:
            True if the Value object is greater than or equal to the other object, False otherwise.
        """

        # Convert 'other' to a Value instance if it's not already and perform the comparison
        other = other if isinstance(other, Value) else Value(other)

        return self.data >= other.data


    def relu(self):

        """
        Apply the rectified linear unit (ReLU) activation function to the Value object.

        Returns:
            A new Value object representing the result of the ReLU activation.
        """

        # Create a new Value object based on the ReLU function applied to the current value
        out = Value(self.data, _op='relu', _prev=[self]) if self.data > 0 else Value(0, _op='relu', _prev=[self])

        # Define the backward function for computing gradients
        def _backward():
            self.grad += out.grad if self.data > 0 else 0

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out


    def log(self):

        """
        Apply the natural logarithm function to the Value object.

        Returns:
            A new Value object representing the result of the logarithm.
        """

        # Calculate the natural logarithm of the current value and create a new Value object
        out = Value(math.log(self.data), _op='log', _prev=[self])

        # Define the backward function for computing gradients
        def _backward():
            self.grad += (1 / self.data) * out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out

    
    def exp(self):

        """
        Apply the exponential function to the Value object.

        Returns:
            A new Value object representing the result of the exponential function.
        """

        # Calculate the exponential function of the current value and create a new Value object
        out = Value(math.exp(self.data), _op='exp', _prev=(self))

        # Define the backward function for computing gradients
        def _backward():
            self.grad += out.data * out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out


    def tanh(self):

        """
        Apply the hyperbolic tangent (tanh) function to the Value object.

        Returns:
            A new Value object representing the result of the hyperbolic tangent function.
        """

        # Calculate the hyperbolic tangent of the current value and create a new Value object
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, _op='tanh', _prev=[self])

        # Define the backward function for computing gradients
        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out


    def sigmoid(self):

        """
        Apply the sigmoid function to the Value object.

        Returns:
            A new Value object representing the result of the sigmoid function.
        """

        # Calculate the sigmoid of the current value and create a new Value object
        data = self.data
        exp = math.exp(-data)
        out = Value(1 / (exp + 1), _op='sigmoid', _prev=[self])

        # Define the backward function for computing gradients
        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad

        # Assign the backward function to the '_backward' attribute of the output Value
        out._backward = _backward

        return out

        
        
        
    def backward(self) :
    
        """
        Perform backpropagation to compute gradients for each Value object in the computation graph.
        """
        
        # implimentation of Topological sort         
        topo = []
        visited = set()

        def topological_sort(v):
          if isinstance(v,Value) and v not in visited:
            visited.add(v)
            for child in v._prev:
              topological_sort(child)
            topo.append(v)
        
        # Topologically sort the nodes in the computation graph
        topological_sort(self)
        
        # Set the gradient of the output Value to 1 (assuming loss function already set)
        self.grad = 1

        # Backpropagate gradients through the computation graph in reverse topological order
        for v in reversed(topo) : 
            v._backward()
        
