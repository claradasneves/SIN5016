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

def cost_function(
        y_true_one_hot: np.array, 
        y_pred: np.array, 
    ):
    """
    Função de custo multiclasse (entropia cruzada)
    """
    # np.clip previne log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    
    loss = -np.mean(np.sum(y_true_one_hot * np.log(y_pred), axis=1))

    return loss

def gradient_cross_entropy(X, w, v, y_one_hot):
    """
    Gradiente da Entropia Cruzada (Negativo da Log-Verossimilhança).
    """

    y_pred = predict(X, w, v)

    return -y_one_hot / (1 + y_pred)

def gradient_descent(
        X, y_one_hot, 
        W_weights : np.array, V_weights :np.array,
        alpha=0.001, 
        tol=0.001,
        n_epochs=1000):
    
    print('Iniciando Otimizador Gradiente Descendente')
    
    W = np.array(W_weights, dtype=float)
    V = np.array(V_weights, dtype=float)
    
    losses = []
    
    for i in range(n_epochs):
        
        N = X.shape[0]
        
        y_pred = predict(X, W, V)
        logit = X.dot(W)
        w_hidden = tanh(logit)
        
        loss = cost_function(y_one_hot, y_pred)
        losses.append(loss)

        grad_erro = (y_pred - y_one_hot)
        
        dEdV = grad_erro.T.dot(w_hidden)

        erro_propagado = grad_erro.dot(V.T)
        
        Delta1 = erro_propagado * dtanh(w_hidden)
        
        dEdW = X.T.dot(Delta1) / N
        
        W -= alpha * dEdW
        V -= alpha * dEdV
        
        # Verifica a convergência baseado na norma dos gradientes
        if np.linalg.norm(dEdW) < tol and np.linalg.norm(dEdV) < tol: 
            print(f'Convergência atingida no Gradiente | Época {i+1}')
            break
            
    losses = np.array(losses)
    print(f'Avg training loss (GD)={np.mean(losses)}')

    return W, V, losses

def fit(
        model,
        X_train, y_train_one_hot, X_test, y_test_one_hot,
        optimizer='GD',
        epochs=1000, 
        learning_rate=0.01,
    ):
    
    print('='*10)
    print(f'Training classification model with {optimizer}')


    M = X_train.shape[1]
    K = y_train_one_hot.shape[1]
    h = 3 # numero de neurônios

    # Inicializa pesos com valores pequenos para estabilidade
    W = np.random.randn(M, h) * 0.001
    V = np.random.randn(h, K) * 0.001
    
    W_final, V_final, history_loss = gradient_descent(
        X_train, y_train_one_hot, W, V, alpha=learning_rate, n_epochs=epochs
    )

    # Avaliação Final
    y_pred_val = model(X_test, W_final, V_final)

    val_loss = cost_function(y_test_one_hot, y_pred_val)
    
    print('val cost function (final)', val_loss)
    print()

    return W_final, V_final, history_loss

def predict(X: np.array, w: np.array, v: np.array):
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