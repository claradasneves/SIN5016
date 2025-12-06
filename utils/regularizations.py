import numpy as np

def regularization(weights, gamma=0.00001, q=1):
    """
    Função abstrata para calcular a regularização. Considera os cenários de Lasso (q=0) e Ridge (q=1)
    """
    return gamma * (np.abs(weights)**q).sum()

def lasso_regularization(cost_function, w):
    """
    Aplica função de custo com regularização Lasso
    """
    return cost_function + regularization(w, q=1)

def ridge_regularization(cost_function, w):
    """
    Aplica função de custo com regularização Ridge
    """
    return cost_function + regularization(w, q=2)

def elastic_net(cost_function, w):
    """
    Elastic Net
    """
    return cost_function + regularization(w, q=1) + regularization(w, q=2)

def gradient_ridge(gradient_cost_function, w, gamma=0.00001):
    """
    Gradiente descendente da cost function com Ridge
    """
    return gradient_cost_function + 2*np.abs(w)*gamma

def gradient_elastic_net(gradient_cost_function, w, gamma=0.00001):
    """
    Gradiente descendente da Elastic Net
    """
    return gradient_cost_function + gamma*(np.sign(w) + 2*np.abs(w))
