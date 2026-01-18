import numpy as np
from models import MLP, RegLog, CNNModel
from utils.activations import tanh, softmax
from utils.data_processing import one_hot_encoding, split_train_test
from utils.loss_functions import entropia_cruzada
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import numpy as _np
import os as _os
import glob as _glob
from tqdm import tqdm
import time
import pandas as pd
import joblib
import imageio.v3 as imageio
from skimage.transform import resize

parser = argparse.ArgumentParser()

parser.add_argument('-m', 
                    '--model', 
                    help='Escolha `mlp` para MLP ou `reglog` para Regressão Logística')
parser.add_argument('-r', 
                    '--regularization', 
                    default='l2',
                    help='Regularização `l1`, `l2`, ou `elastic-net`')
parser.add_argument('-o', 
                    '--optimizer', 
                    default='gd',
                    help='Otimização `gd` para Gradiente Descendente ou `newton` para Gauss-Newton')
parser.add_argument('--features-dir', 
                    default=None,
                    help='Pasta contendo features HOG (.npy por imagem) ou contendo um único arquivo .npy com features (necessita mapping). Ex: hog_features_2025...')
parser.add_argument('--identity-file', 
                    default=_os.path.join('data', 'atributos', 'identity_CelebA.txt'),
                    help='Arquivo txt com mapeamento imagem -> id (padrão: data/atributos/identity_CelebA.txt)')
parser.add_argument('--num-classes', 
                    type=int, 
                    default=500,
                    help='Número de classes (pessoas) a selecionar. Serão selecionadas as classes com mais amostras. Use 0 para incluir todas (padrão: 500).')
parser.add_argument('--use-attributes', 
                    action='store_true',
                    help='Se ativado, concatena atributos do CelebA às features HOG')
parser.add_argument('--attr-file', 
                    default=_os.path.join('data', 'atributos', 'list_attr_celeba.csv'),
                    help='CSV com atributos do CelebA (usado apenas se --use-attributes)')
parser.add_argument('--experiment_name',
                    help='Nome do experimento para salvar em logs')
parser.add_argument('--train-on-images',
                    action='store_true',
                    help='Se ativado, carrega e treina com imagens em vez de features HOG')
parser.add_argument('--images-dir',
                    default=None,
                    help='Diretório contendo imagens em estrutura aninhada (classe_id/imagem.jpg). Ex: /Users/claradasneves/Documents/selecionadas')
parser.add_argument('--img-size',
                    type=int,
                    default=64,
                    help='Tamanho para redimensionar imagens (padrão: 64)')
parser.add_argument('--batch-size',
                    type=int,
                    default=32,
                    help='Batch size para treinamento da CNN (padrão: 32)')

args = parser.parse_args()


def _load_identity(identity_file: str):
    import pandas as pd
    # tenta inferir separador (alguns arquivos têm múltiplos espaços)
    df = pd.read_csv(identity_file, sep=r'\s+', header=None, names=['imagem', 'id'], engine='python')
    return df

def _load_attributes(attr_file: str):
    import pandas as pd

    df = pd.read_csv(attr_file)

    # garantir coluna de imagem
    if 'image_id' in df.columns:
        df.rename(columns={'image_id': 'imagem'}, inplace=True)

    # converter -1 / 1 para 0 / 1
    attr_cols = df.columns.drop('imagem')
    df[attr_cols] = (df[attr_cols] == 1).astype(int)

    df['basename'] = df['imagem'].apply(lambda s: _os.path.basename(s))
    df = df.set_index('basename')

    return df, attr_cols.tolist()

def _load_features_and_labels_from_dir(features_dir: str, identity_df):
    """
    Tenta carregar features a partir de um diretório.

    Suporta dois casos comuns:
    1) Diretório contendo vários arquivos .npy, cada um correspondendo a uma imagem. O nome do arquivo
       pode ser o stem da imagem (ex: '000002.npy') ou o nome completo sem extensão (ex: '000002.jpg.npy').
    2) Diretório contendo um único arquivo .npy com shape (N, D) -- nesse caso precisamos de um arquivo
       de mapeamento; se não houver, o loader retorna erro informativo.

    Retorna (X, y, image_names)
    """
    features_dir = _os.path.abspath(features_dir)
    npy_files = sorted([p for p in _glob.glob(_os.path.join(features_dir, '*.npy'))])

    # preparar mapping de identidade: imagem (basename) -> pessoa id
    identity_df = identity_df.copy()
    identity_df['basename'] = identity_df['imagem'].apply(lambda s: _os.path.basename(s))
    identity_df['stem'] = identity_df['basename'].apply(lambda s: _os.path.splitext(s)[0])

    # Caso A: vários .npy por imagem
    if len(npy_files) > 1:
        # construir dict stem->path e basename->path
        stem_map = {}
        basename_map = {}
        for p in npy_files:
            name = _os.path.basename(p)
            stem = _os.path.splitext(name)[0]
            # aceitar sufixos comuns como '_hog' -> mapear também para o prefixo
            stem_map[stem] = p
            if stem.endswith('_hog'):
                stem_map[stem[:-4]] = p
            basename_map[name] = p

        X_list = []
        y_list = []
        img_names = []
        for _, row in identity_df.iterrows():
            # prefer stem match
            if row['stem'] in stem_map:
                feat_path = stem_map[row['stem']]
            elif row['basename'] in basename_map:
                feat_path = basename_map[row['basename']]
            else:
                continue

            try:
                feat = _np.load(feat_path)
            except Exception:
                # pular arquivo com erro
                continue

            X_list.append(feat.reshape(-1))
            y_list.append(int(row['id']))
            img_names.append(row['basename'])

        if not X_list:
            raise RuntimeError('Nenhuma feature compatível encontrada no diretório fornecido.')

        X = _np.vstack(X_list)
        y = _np.array(y_list, dtype=int)
        return X, y, img_names

    # Caso B: único .npy no diretório
    if len(npy_files) == 1:
        # tenta carregar e tentar alinhar pelo número de entradas
        feats = _np.load(npy_files[0])
        # se há arquivo de mapping (image_names.txt, paths.npy etc.)
        mapping_candidates = [
            _os.path.join(features_dir, 'image_names.txt'),
            _os.path.join(features_dir, 'image_names.npy'),
            _os.path.join(features_dir, 'paths.txt'),
            _os.path.join(features_dir, 'paths.npy'),
        ]
        for m in mapping_candidates:
            if _os.path.exists(m):
                # carregar mapeamento
                if m.endswith('.txt'):
                    with open(m, 'r') as fh:
                        names = [ln.strip() for ln in fh if ln.strip()]
                else:
                    names = list(_np.load(m))

                # alinhar names com identity
                df_map = identity_df[identity_df['basename'].isin([_os.path.basename(n) for n in names])]
                if df_map.empty:
                    raise RuntimeError('Arquivo de mapping encontrado, mas nenhuma imagem bate com identity file.')

                # construir X,y filtrando apenas names que estão no identity
                X_list = []
                y_list = []
                img_names = []
                for i, n in enumerate(names):
                    bn = _os.path.basename(n)
                    rows = identity_df[identity_df['basename'] == bn]
                    if rows.empty:
                        continue
                    X_list.append(feats[i].reshape(-1))
                    y_list.append(int(rows.iloc[0]['id']))
                    img_names.append(bn)

                if not X_list:
                    raise RuntimeError('Nenhuma correspondência entre mapping e identity file.')

                X = _np.vstack(X_list)
                y = _np.array(y_list, dtype=int)
                return X, y, img_names

        # sem mapping: se o número de features coincide com a quantidade de linhas no identity_df,
        # assumimos que a ordem corresponde (aviso será emitido)
        if feats.shape[0] == identity_df.shape[0]:
            X = feats
            y = identity_df['id'].to_numpy(dtype=int)
            img_names = identity_df['basename'].tolist()
            print('Aviso: alinhando features por posição com identity file (mesmo número de entradas).')
            return X, y, img_names

        raise RuntimeError('Não foi possível inferir mapeamento entre features e identity file. Forneça um diretório com .npy por imagem nomeados ou um arquivo de mapping.')

def stratified_split(X, y, train_size=0.9):
    """Divide X e y mantendo a proporção de classes em ambos conjuntos (estratificação).
    
    Args:
        X: features (N, D)
        y: one-hot encoded labels (N, K)
        train_size: proporção para treino (padrão 0.9 = 90% treino, 10% teste)
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    N = X.shape[0]
    y_idx = _np.argmax(y, axis=1)  # converter one-hot para índices
    unique_classes = _np.unique(y_idx)
    
    train_idx = []
    test_idx = []
    
    for cls in unique_classes:
        cls_indices = _np.where(y_idx == cls)[0]
        n_cls = len(cls_indices)
        n_train_cls = max(1, int(n_cls * train_size))
        
        # embaralhar índices dessa classe
        _np.random.shuffle(cls_indices)
        
        train_idx.extend(cls_indices[:n_train_cls])
        test_idx.extend(cls_indices[n_train_cls:])
    
    train_idx = _np.array(train_idx)
    test_idx = _np.array(test_idx)
    
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]

def plot_kfold_losses(all_folds_history, title):
    # 1. Padronizar o comprimento (folds podem ter parado antes devido ao 'tol')
    # Encontramos o número máximo de épocas executadas
    max_epochs = max(len(fold) for fold in all_folds_history)
    
    # Criamos matrizes preenchidas com NaN para acomodar tamanhos diferentes
    train_losses = np.full((len(all_folds_history), max_epochs), np.nan)
    val_losses = np.full((len(all_folds_history), max_epochs), np.nan)

    for i, fold in enumerate(all_folds_history):
        fold_arr = np.array([fold])

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

    plt.title(title)
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    save_title = title.split('\n')[0]
    # plt.savefig(f'experiments/fig/{save_title}.png')

    plt.show()

def plot_cnn_kfold_losses(history, title):
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss values
    # plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # plt.tight_layout()
    save_title = title.split('\n')[0]
    plt.show()
    plt.savefig(f'./experiments/fig/{save_title}.png')


def normalize_features(X):
    X = X.astype(float)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def add_attributes(X, y, image_names, attr_file):
    print('loading attributes...')

    attr_df, attr_cols = _load_attributes(attr_file)

    # cria Series com nomes das imagens e seus índices originais
    img_series = pd.Series(np.arange(len(image_names)), index=image_names)

    # interseção entre imagens e atributos
    common_imgs = img_series.index.intersection(attr_df.index)

    if len(common_imgs) == 0:
        raise RuntimeError('Nenhum atributo pôde ser alinhado com as imagens.')

    # índices válidos em X e y
    valid_idx = img_series.loc[common_imgs].values

    # atributos alinhados
    A = attr_df.loc[common_imgs, attr_cols].to_numpy()

    X = X[valid_idx]
    y = y[valid_idx]

    X = np.hstack([X, A])

    print(f'Usando HOG ({X.shape[1] - A.shape[1]}) + atributos ({A.shape[1]})')

    return X, y

def remap_and_filter_classes(X, y_raw, num_classes_target):
    print('remapping and filtering...')
    # mapear ids originais → labels 0..K-1
    unique_ids = np.unique(y_raw)
    id_to_label = {int(v): i for i, v in enumerate(unique_ids)}
    y_idx = np.array([id_to_label[int(v)] for v in y_raw], dtype=int)

    # filtrar top-N classes
    if num_classes_target > 0 and num_classes_target < len(unique_ids):
        class_counts = np.bincount(y_idx)
        top_classes = np.argsort(class_counts)[-num_classes_target:][::-1]

        mask = np.isin(y_idx, top_classes)
        X = X[mask]
        y_idx = y_idx[mask]

        remap = {c: i for i, c in enumerate(top_classes)}
        y_idx = np.array([remap[v] for v in y_idx], dtype=int)

        print(f'Selecionadas top {num_classes_target} classes.')

    return X, y_idx

def load_mock_dataset():
    mock_faces = fetch_olivetti_faces()

    X = mock_faces.data
    y_idx = mock_faces.target

    N = X.shape[0]
    X = X.reshape(N, -1)

    num_classes = len(_np.unique(y_idx))
    num_features = X.shape[1]

    # num_classes = len(np.unique(y_idx))
    # num_features = X.shape[1]

    y = one_hot_encoding(y_idx, num_classes)

    return X.astype(float), y, num_features, num_classes

def load_images_from_nested_structure(root_dir, img_size=64, max_classes=None):
    print(f"Carregando imagens de {root_dir}...")
    
    class_dirs = sorted([d for d in _glob.glob(_os.path.join(root_dir, '*')) if _os.path.isdir(d)])
    
    if max_classes:
        class_dirs = class_dirs[:max_classes]
    
    images = []
    labels = []
    class_names = []
    
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Carregando classes")):
        class_name = _os.path.basename(class_dir)
        class_names.append(class_name)
        
        image_files = _glob.glob(_os.path.join(class_dir, '*.jpg')) + \
                      _glob.glob(_os.path.join(class_dir, '*.png'))
        
        for img_file in image_files:
            try:
                img = imageio.imread(img_file)
                
                # lida com escalas de cinza
                if len(img.shape) == 2:
                    img = _np.stack([img] * 3, axis=-1)
                # imagens coloridas
                elif img.shape[2] == 4:
                    img = img[:, :, :3]
                
                # Resize
                img = resize(img, (img_size, img_size, 3), anti_aliasing=True)
                
                images.append(img.astype(_np.float32))
                labels.append(class_idx)
            except Exception as e:
                pass 
    
    X = _np.array(images)
    y_idx = _np.array(labels)
    
    print(f"Carregadas {len(images)} imagens de {len(class_names)} classes")
    print(f"Shape dos dados: {X.shape}")
    
    num_classes = len(class_names)
    # CNN expects one-hot encoding
    y = one_hot_encoding(y_idx, num_classes)
    
    sample_shape = X.shape[1:]
    
    return X, y, sample_shape, num_classes

def load_celeba_dataset(args):
    print('loading celebA...')
    id_df = _load_identity(args.identity_file)
    X_raw, y_raw, image_names = _load_features_and_labels_from_dir(
        args.features_dir, id_df
    )

    # X = normalize_features(X_raw)
    X = X_raw.astype(float)

    if args.use_attributes:
        X, y_raw = add_attributes(X, y_raw, image_names, args.attr_file)

    X, y_idx = remap_and_filter_classes(X, y_raw, args.num_classes)

    num_classes = len(np.unique(y_idx))
    y = one_hot_encoding(y_idx, num_classes)

    return X.astype(float), y, X.shape[1], num_classes

def load_dataset(args):
    if args.train_on_images and args.images_dir:
        return load_images_from_nested_structure(
            args.images_dir,
            img_size=args.img_size,
            max_classes=args.num_classes if args.num_classes > 0 else None
        )
    elif args.features_dir:
        return load_celeba_dataset(args)
    else:
        return load_mock_dataset()

def build_model(args, num_features, num_classes):
    if args.model is None:
        raise ValueError("Use --model mlp, reglog ou cnn")

    # Se está usando imagens, force CNN
    if args.train_on_images and isinstance(num_features, tuple):
        if args.model != 'cnn':
            raise ValueError("Quando usando --train-on-images, deve-se usar --model cnn")
        return CNNModel(
            num_classes=num_classes,
            input_shape=num_features,
        )

    if args.model == 'mlp':
        return MLP(
            num_features=num_features,
            num_classes=num_classes,
            num_neurons=64,
            hidden_layer_activation=tanh,
            output_layer_activation=softmax,
            cost_function=entropia_cruzada,
            regularization=args.regularization,
        )

    if args.model == 'reglog':
        return RegLog(
            num_features=num_features,
            num_classes=num_classes,
            cost_function=entropia_cruzada,
            regularization=args.regularization,
        )

    if args.model == 'cnn':
        return CNNModel(
            num_classes=num_classes,
            input_shape=num_features,
        )

def evaluate_model(model, X_test, y_test, top_k=10):
    y_pred = model.predict(X_test)

    # Detectar formato: one-hot (2D) ou indices (1D)
    if y_test.ndim == 2:
        # one-hot encoded
        y_test_idx = np.argmax(y_test, axis=1)
    else:
        # numeric indices
        y_test_idx = y_test

    # acurácia global (y_pred é probabilidades)
    acc = np.mean(
        np.argmax(y_pred, axis=1) == y_test_idx
    )

    print(f"Test accuracy (global): {acc:.4f}")

    # acurácia por classe
    acc_per_class = accuracy_per_class(y_test_idx, y_pred)
    acc_values = np.array([v for v in acc_per_class.values() if not np.isnan(v)])

    print(f"Acurácia média por classe: {acc_values.mean():.4f}")
    print(f"Desvio padrão: {acc_values.std():.4f}")

    print_top_k_classes(acc_per_class, k=top_k)

    return acc, acc_per_class

def print_top_k_classes(acc_per_class, k=10):
    items = [(c, a) for c, a in acc_per_class.items() if not np.isnan(a)]
    items.sort(key=lambda x: x[1])

    print(f"\nTop {k} piores classes:")
    for c, a in items[:k]:
        print(f"Classe {c}: {a:.3f}")

    print(f"\nTop {k} melhores classes:")
    for c, a in items[-k:]:
        print(f"Classe {c}: {a:.3f}")

def accuracy_per_class(y_true, y_pred):
    """
    y_true: numeric indices
    y_pred: class probabilities (softmax output)
    """
    y_true_idx = y_true  # já são indices numéricos
    y_pred_idx = np.argmax(y_pred, axis=1)

    num_classes = y_pred.shape[1]
    acc_per_class = {}

    for c in range(num_classes):
        idx = (y_true_idx == c)
        if np.sum(idx) == 0:
            acc_per_class[c] = np.nan
        else:
            acc_per_class[c] = np.mean(y_pred_idx[idx] == c)

    return acc_per_class

def main(args):
    EPOCHS = 500
    LEARNING_RATE = 1e-1

    # 1 - carregar dados
    result = load_dataset(args)
    X, y, num_features, num_classes = result

    # 2 - split
    if args.train_on_images:
        # For images, convert one-hot to indices for stratify
        y_idx = np.argmax(y, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y_idx
        )
    else:
        X_train, y_train, X_test, y_test = split_train_test(X, y, rate=0.7)

    # 3 - normalize
    if args.train_on_images:
        mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        std = X_train.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    else:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test  = (X_test - mean) / std

    print('Train set shapes', X_train.shape, y_train.shape)
    print('Test set shapes', X_test.shape, y_test.shape)

    # 4 - modelo
    model = build_model(args, num_features, num_classes)

    # 5 - treino
    if args.train_on_images and args.model == 'cnn':
        # Para CNN com images
        history_loss, test_loss = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            learning_rate=1e-3,
            epochs=3,
            batch_size=args.batch_size,
            validation_split=0.1
        )
    else:
        # Para MLP/RegLog com features
        history_loss, test_loss = model.fit(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            optimizer=args.optimizer,
        )

    # 6 - avaliação
    test_acc, acc_per_class = evaluate_model(model, X_test, y_test)

    # 7 - plot
    regularization_str = getattr(model, 'regularization', 'NA')
    title = \
        model.__class__.__name__+ f'+HOG={args.use_attributes}-' + \
            f"(regularizer={regularization_str}_optimizer={args.optimizer})" \
            f"\n test loss={round(test_loss, 3)}" \
            f", test acc={round(test_acc, 3)}"
    
    plot_cnn_kfold_losses(
        history_loss,
        title=title,
    )

    # 8 - salva modelo c/ os melhores pesos 
    model_name = title.split('\n')[0]
    joblib.dump(model, f'experiments/weights/{model_name}.joblib')

if __name__ == "__main__":
    main(args)