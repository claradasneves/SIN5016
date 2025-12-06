import pandas as pd
import numpy as np
import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

def one_hot_encoding(y, K):
    """
    Aplica transformação na representação categórica dos rótulos em representação numérica
    
    Args:
        :param y: lista de rótulos com representação categórica
        :param K: quantidade de classes

    Returns
        lista de classes transformada em representação numérica
    """
    N = y.shape[0]
    y_one_hot = np.zeros((N, K))
    y_one_hot[np.arange(N), y] = 1
    
    return y_one_hot

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


def extract_hog_feature(
        image: np.ndarray,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
        visualize: bool = False,
        resize_to: tuple = None,
    ):
    """
    Extrai o vetor de características HOG de uma única imagem.

    Args:
        image: numpy array da imagem (H,W) ou (H,W,3)
        orientations: número de orientações (bins)
        pixels_per_cell: tamanho (em pixels) de cada célula
        cells_per_block: número de células por bloco
        visualize: se True retorna também a imagem HOG para visualização
        resize_to: tupla (new_h, new_w) para redimensionar a imagem antes de extrair

    Returns:
        se visualize==False: array 1D com as features HOG
        se visualize==True: (features, hog_image)
    """

    if resize_to is not None:
        image = resize(image, resize_to, anti_aliasing=True)

    # garante escala de cinza
    if image.ndim == 3:
        image_gray = rgb2gray(image)
    else:
        image_gray = image

    if visualize:
        features, hog_image = hog(
            image_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True,
        )
        return features, hog_image
    else:
        features = hog(
            image_gray,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            feature_vector=True,
        )
        return features


def extract_hog_batch(
        images: list,
        orientations: int = 9,
        pixels_per_cell: tuple = (16, 16),
        cells_per_block: tuple = (2, 2),
        resize_to: tuple = None,
    ):
    """
    Extrai features HOG de uma lista/iterável de imagens em numpy arrays.

    Returns:
        numpy.array de shape (N, D) onde D é o tamanho do vetor HOG
    """
    feats = []
    for img in images:
        f = extract_hog_feature(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=True,
            resize_to=resize_to,
        )
        feats.append(f)

    return np.array(feats)


def extract_hog_from_paths(
        paths: list,
        orientations: int = 9,
        pixels_per_cell: tuple = (16, 16),
        cells_per_block: tuple = (2, 2),
        resize_to: tuple = None,
    ):
    """
    Carrega imagens de uma lista de paths e extrai features HOG.

    Args:
        paths: lista de caminhos para arquivos de imagem

    Returns:
        numpy.array com as features HOG (N, D)
    """
    images = []
    for p in paths:
        img = imageio.imread(p)
        images.append(img)

    return extract_hog_batch(
        images,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        resize_to=resize_to,
    )