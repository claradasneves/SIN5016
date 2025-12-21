import numpy as np

def regularization(weights, gamma=1e-5, q=1):
    """
    Função abstrata para calcular a regularização. Considera os cenários de Lasso (q=0) e Ridge (q=1)
    """
    return gamma * (np.abs(weights)**q).sum()

def lasso(weights, coef=0.01, bias=True):
    """
    Aplica função de custo com regularização Lasso
    """
    if bias:
        weights[0, :] = 0

    return (coef * weights.shape[0]) * np.sign(weights)

def ridge(weights, coef=0.1, bias=True):
    """
    Aplica função de custo com regularização Ridge
    """
    reg = (coef / weights.shape[0]) * weights.copy()

    if bias:
        reg[0, :] = 0.0  # não regulariza bias

    return reg

def elastic_net(weights, coef=0.01, bias=True):
    """
    Elastic Net
    """
    if bias:
        weights[0, :] = 0

    l1 = np.sign(weights)
    l2 = weights

    return (coef / weights.shape[0]) * (l1 + l2)

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
