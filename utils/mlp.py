import numpy as np


def tanh(z):
    """Função da tangente hiperbólica"""
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def dtanh(z):
    """derivada da função da tangente hiperbólica"""
    # z já é a tanh
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

def mlp(X: np.array, w: np.array, v: np.array):
    """
    Perceptron multicamada

    Args
        X: matriz com os dados
        w: matriz com pesos da camada de entrada
        v: matriz com pesos da camada de saida

    Returns
        lista de rótulos preditos
    """
    
    logit = X.dot(w)

    z_hidden = tanh(logit)

    z_hidden = z_hidden.dot(v) # isso pq estou em um problema multi-classe
    # se binário, eu usaria o hadamard-product

    return softmax(z_hidden)