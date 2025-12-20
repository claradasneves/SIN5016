import numpy as np
from models import MLP, RegLog
from utils.activations import tanh, softmax
from utils.data_processing import one_hot_encoding, split_train_test
from utils.loss_functions import entropia_cruzada
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-m', '--model', help='Escolha `mlp` para MLP ou `reglog` para Regressão Logística')
parser.add_argument('-r', '--regularization', help='Regularização `l1`, `l2`, ou `elastic-net`')

args = parser.parse_args()

def plot_kfold_losses(all_folds_history):
    # 1. Padronizar o comprimento (folds podem ter parado antes devido ao 'tol')
    # Encontramos o número máximo de épocas executadas
    max_epochs = max(len(fold) for fold in all_folds_history)
    
    # Criamos matrizes preenchidas com NaN para acomodar tamanhos diferentes
    train_losses = np.full((len(all_folds_history), max_epochs), np.nan)
    val_losses = np.full((len(all_folds_history), max_epochs), np.nan)

    for i, fold in enumerate(all_folds_history):
        fold_arr = np.array(fold)
        train_losses[i, :len(fold)] = fold_arr[:, 0]
        val_losses[i, :len(fold)] = fold_arr[:, 1]

    # 2. Calcular a média ignorando os NaNs (onde os folds já tinham convergido)
    mean_train = np.nanmean(train_losses, axis=0)
    mean_val = np.nanmean(val_losses, axis=0)
    
    # 3. Calcular o Desvio Padrão (para mostrar a variabilidade entre folds)
    std_train = np.nanstd(train_losses, axis=0)
    std_val = np.nanstd(val_losses, axis=0)

    epochs_range = np.arange(max_epochs)

    plt.figure(figsize=(10, 6))

    # Plot Treino
    plt.plot(epochs_range, mean_train, label='Treino (Média)', color='blue', lw=2)
    plt.fill_between(epochs_range, mean_train - std_train, mean_train + std_train, color='blue', alpha=0.2)

    # Plot Validação
    plt.plot(epochs_range, mean_val, label='Validação (Média)', color='red', lw=2)
    plt.fill_between(epochs_range, mean_val - std_val, mean_val + std_val, color='red', alpha=0.2)

    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == '__main__':
    # 1 - carrega dados
    EPOCHS = 50
    
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
    regularization = args.regularization

    if args.model == 'mlp': 
        model = MLP(
            num_features=num_features,
            num_classes=num_classes,
            num_neurons=1,
            hidden_layer_activation=tanh,
            output_layer_activation=softmax,
            cost_function=entropia_cruzada,
        )

    elif args.model == 'reglog':
        model = RegLog(
            num_features=num_features,
            num_classes=num_classes,
            cost_function=entropia_cruzada,
            regularization=regularization,
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

    # plt.plot(range(EPOCHS), np.mean(history_loss[:, :, 0]), label='train')
    # plt.plot(range(EPOCHS), np.mean(history_loss[:, :, 1]), label='val')
    # plt.legend()
    # plt.show()
    plot_kfold_losses(history_loss)
