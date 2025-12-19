"""
Docstring for utils.loss_functions

Funções de custo para os modelos
"""

import numpy as np

def mae(y_true: np.array, y_pred: np.array, gradient=False):
    
    if not gradient:
        return np.mean(np.abs(y_pred - y_true))
    
    else:
        # Gradiente descendente da MAE
        return np.sign(y_pred - y_true)

def mse(y_true: np.array, y_pred: np.array, gradient=False):   
    
    if not gradient:
        return np.mean((y_pred - y_true)**2)
    
    else:
        # Gradiente descendente da MSE
        return 2*np.mean(y_pred - y_true)

def entropia_cruzada(
        y_true: np.array, 
        y_pred: np.array, 
    ):
    """
    Função de custo multiclasse (entropia cruzada)
    """
    # np.clip previne log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    return loss

def gradient_cross_entropy(y_true: np.array, y_pred: np.array):
    """
    Gradiente da Entropia Cruzada (Negativo da Log-Verossimilhança).
    """
    return -y_true / (1 + y_pred)

