"""
Funções de ativações úteis para os modelos
"""

import numpy as np

def tanh(z, derivative=False):
    """Função da tangente hiperbólica"""
    
    if not derivative:
        # tanh
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    else:
        # derivada da tanh
        # z já é a tanh
        z = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
        return 1 - z**2

def softmax(logit):
    """
    Função de ativação simples (softmax pq temos um problema multiclasse)
    """
    # estabilização para evitar explosão
    logit_max = np.max(logit, axis=1, keepdims=True)
    logit_estavel = logit - logit_max
    
    exp_logit = np.exp(logit_estavel)

    exp_categorias = np.sum(exp_logit, axis=1, keepdims=True)
    
    return exp_logit / exp_categorias