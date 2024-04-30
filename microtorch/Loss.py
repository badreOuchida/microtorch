import numpy as np

def MSELoss(y_true, y_pred):

    """
    Calculate the mean squared error (MSE) loss.

    Args:
        y_true (numpy.ndarray): Array containing the true labels.
        y_pred (numpy.ndarray): Array containing the predicted values.

    Returns:
        float: The mean squared error (MSE).
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    loss = np.sum((y_true - y_pred) ** 2) / len(y_true)

    return loss

def MSELossL2(y_true, y_pred, parameters, alpha):
    
    """
    Compute the Mean Squared Error (MSE) loss augmented with L2 regularization.

    Args:
        y_true (numpy.ndarray): The true values or ground truth.
        y_pred (numpy.ndarray): The predicted values.
        parameters (callable): A function that provides the model parameters.
        alpha (float): The regularization parameter, controlling the strength of the L2 regularization.

    Returns:
        float: The combined loss value, which is the sum of the MSE loss and the L2 regularization term.
    """

    mse_loss = MSELoss(y_true, y_pred)

    l2_regularization_term = alpha * sum(p ** 2 for p in parameters())

    combined_loss = mse_loss + l2_regularization_term

    return combined_loss
