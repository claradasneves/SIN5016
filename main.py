import numpy as np
from utils.models import MLP, RegLog
from utils.activations import tanh, softmax
from utils.data_processing import one_hot_encoding, split_train_test
from utils.loss_functions import entropia_cruzada
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 1 - carrega dados
    EPOCHS = 500
    # N = 1000
    # height = 128
    # widht = 128
    # channels = 3

    # X = np.random.rand(N, channels, height, widht) # imagens mock
    # y = np.random.randint(low=0, high=2, size=N)
    
    mock_faces = fetch_olivetti_faces()
    X = mock_faces.data
    y = mock_faces.target

    N = X.shape[0]
    K=len(np.unique(mock_faces.target))

    # 2 - converte rótulos com one-hot-encoding
    X = X.reshape(N, -1) # shape: (N, channels * height * width)

    X /= 255
    # TODO X = np.insert(X, 0, 1, axis=1) # add bias
    
    y = one_hot_encoding(y, K)

    # 3 - divide em partição de treino/teste
    X_train, y_train, X_test, y_test = split_train_test(X, y, rate=0.9)
    
    print(X_train.shape, y_train.shape)

    # 4 - define modelo
    model = MLP(
        num_features=X.shape[-1],
        num_classes=K,
        num_neurons=10,
        hidden_layer_activation=tanh,
        output_layer_activation=softmax,
    )

    # 5 - aplica .fit() do modelo
    history_loss = model.fit(
        cost_function=entropia_cruzada,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=1e-5,
        epochs=EPOCHS,
    )

    print('avg training loss:', np.mean(history_loss))

    plt.plot(range(EPOCHS), history_loss[:, 0], label='train')
    plt.plot(range(EPOCHS), history_loss[:, 1], label='val')
    plt.legend()
    plt.show()

