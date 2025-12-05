import pandas as pd
import numpy as np

def split_train_test(X, y, rate=0.8, shuffle=True):
    """
    Divide o dataset em conjuntos de treino e teste. Segue a proporção 80/20, mas é ajustável.

    Args:
        X: pandas Dataframe com o conjunto de dados
        y; lista de rótulos verdadeiros
        rate: percentual a compor a partição de treino
        shuffle: indicador para embaralhar a separação das partições

    Returns
        X_train_set: dados do conjunto da partição de treino
        y_train_set: rotulos verdadeiros da partição de treino
        X_test_set: dados do conjunto da partição de teste
        y_test_set: rotulos verdadeiros da partição de teste

    """

    N = X.shape[0]

    if shuffle:
        idx = np.random.permutation(N)
        
        if isinstance(X, pd.DataFrame):
            X = X.iloc[idx]
        else:
            X = X[idx]
        y = y[idx]

    train_len = round(rate * N)
    test_len = N - train_len

    X_train_set = X[0:train_len]
    y_train_set = y[0:train_len]

    X_test_set = X[train_len: train_len+test_len]
    y_test_set = y[train_len: train_len+test_len]
    

    if isinstance(X_train_set, pd.DataFrame):
        X_train_set = X_train_set.to_numpy()
        X_test_set = X_test_set.to_numpy()

    return (X_train_set, y_train_set, 
            X_test_set, y_test_set)