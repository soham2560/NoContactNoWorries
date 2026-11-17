# optim_loss.py

import torch.nn as nn
import torch.optim as optim

def get_loss(loss_name: str, **kwargs):
    """
    Returns a loss function instance based on the loss_name.
    
    Supported loss options:
      - "crossentropy": nn.CrossEntropyLoss (for multi-class classification)
      - "mse": nn.MSELoss (for regression tasks)
      - "l1": nn.L1Loss (for regression, less sensitive to outliers)
    
    Args:
        loss_name (str): Identifier for the loss function.
        **kwargs: Additional keyword arguments for the loss constructor.
    
    Returns:
        nn.Module: A PyTorch loss function.
    """
    loss_name = loss_name.lower()
    if loss_name == 'crossentropy':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'mse':
        return nn.MSELoss(**kwargs)
    elif loss_name == 'l1':
        return nn.L1Loss(**kwargs)
    else:
        raise ValueError(f"Loss function '{loss_name}' is not supported.")

def get_optimizer(optimizer_name: str, model_parameters, lr: float, **kwargs):
    """
    Returns an optimizer instance based on the optimizer_name.
    
    Supported optimizer options:
      - "adam": optim.Adam
      - "sgd": optim.SGD (optionally with momentum)
      - "rmsprop": optim.RMSprop
    
    Args:
        optimizer_name (str): Identifier for the optimizer.
        model_parameters: The parameters of the model (model.parameters()).
        lr (float): Learning rate.
        **kwargs: Additional keyword arguments for the optimizer.
    
    Returns:
        torch.optim.Optimizer: A PyTorch optimizer.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model_parameters, lr=lr, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model_parameters, lr=lr, **kwargs)
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' is not supported.")

def get_scheduler(scheduler_name: str, optimizer, **kwargs):
    """
    Returns a learning rate scheduler instance based on the scheduler_name.
    
    Supported scheduler options:
      - "steplr": optim.lr_scheduler.StepLR
      - "exponentiallr": optim.lr_scheduler.ExponentialLR
    
    Args:
        scheduler_name (str): Identifier for the scheduler.
        optimizer: The optimizer to which the scheduler will be applied.
        **kwargs: Additional keyword arguments for the scheduler.
    
    Returns:
        A PyTorch learning rate scheduler.
    """
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'steplr':
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == 'exponentiallr':
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' is not supported.")
