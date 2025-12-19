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
        resize_to: tuple = (128, 128),
    ):
    """
    Extrai o vetor de características HOG de uma única imagem.
    Imagens são redimensionadas para tamanho fixo e normalizadas
    para reduzir variação de iluminação.
    """

    # resize fixo
    image = resize(image, resize_to, anti_aliasing=True)

    # garante escala de cinza
    if image.ndim == 3:
        image_gray = rgb2gray(image)
    else:
        image_gray = image.astype(np.float32)

    # normalização de iluminação
    mean = image_gray.mean()
    std = image_gray.std() + 1e-8
    image_gray = (image_gray - mean) / std

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
            visualize=False,
            feature_vector=True,
        )
        return features


def extract_hog_batch(
        images: list,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
    ):
    """
    Extrai features HOG de uma lista de imagens já alinhadas.
    Todas são normalizadas e redimensionadas para 128x128.
    """
    feats = []
    for img in images:
        f = extract_hog_feature(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
        )
        feats.append(f)

    return np.array(feats)


def extract_hog_from_paths(
        paths: list,
        orientations: int = 9,
        pixels_per_cell: tuple = (8, 8),
        cells_per_block: tuple = (2, 2),
    ):
    images = []
    for p in paths:
        img = imageio.imread(p)
        images.append(img)

    return extract_hog_batch(
        images,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
    )