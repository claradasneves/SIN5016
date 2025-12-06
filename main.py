import numpy as np
from utils.models import MLP
from utils.activations import tanh, softmax
from utils.data_processing import one_hot_encoding, split_train_test
from utils.loss_functions import entropia_cruzada

if __name__ == '__main__':
    # 1 - carrega dados
    X = np.random.rand(5, 300)
    y = np.array([1, 1, 0, 0, 1])
    
    # 2 - converte rótulos com one-hot-encoding
    y = one_hot_encoding(y, 2)

    # 3 - divide em partição de treino/teste
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    
    print(X_train.shape, y_train.shape)

    # 4 - define modelo
    model = MLP(
        num_features=X.shape[-1],
        num_classes=2,
        num_neurons=3,
        hidden_layer_activation=tanh,
        output_layer_activation=softmax,
    )

    # 5 - aplica .fit() do modelo
    training_loss = model.fit(
        cost_function=entropia_cruzada,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )

    print('avg training loss:', np.mean(training_loss))

