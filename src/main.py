import numpy as np
from .models import MLP, RegLog
from .utils.activations import tanh, softmax
from .utils.data_processing import one_hot_encoding, split_train_test
from .utils.loss_functions import entropia_cruzada
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1 - carrega dados
    EPOCHS = 1000
    
    mock_faces = fetch_olivetti_faces()
    
    X = mock_faces.data
    y = mock_faces.target


    # 2 - converte rótulos com one-hot-encoding
    N = X.shape[0]
    X = X.reshape(N, -1) # reshape -> (N, channels * height * width)
    
    num_classes=len(np.unique(mock_faces.target))
    num_features = X.shape[-1]

    y = one_hot_encoding(y, num_classes)
    
    # 2.1 - normaliza imagens
    X /= 255
    
    # 3 - divide em partição de treino/teste
    X_train, y_train, X_test, y_test = split_train_test(X, y, rate=0.9)
    
    print('Train set shapes', X_train.shape, y_train.shape)
    print('Test set shapes', X_test.shape, y_test.shape)

    # 4 - define modelo
    # model = MLP(
    #     num_features=num_features,
    #     num_classes=num_classes,
    #     num_neurons=1,
    #     hidden_layer_activation=tanh,
    #     output_layer_activation=softmax,
    #     cost_function=entropia_cruzada,
    # )

    model = RegLog(
        num_features=num_features,
        num_classes=num_classes,
        cost_function=entropia_cruzada,
        regularization='l1',
    )

    # 5 - aplica .fit() do modelo
    history_loss = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=1e-4,
        epochs=EPOCHS,
    )

    print('avg training loss:', np.mean(history_loss))

    plt.plot(range(EPOCHS), history_loss[:, 0], label='train')
    plt.plot(range(EPOCHS), history_loss[:, 1], label='val')
    plt.legend()
    plt.show()

