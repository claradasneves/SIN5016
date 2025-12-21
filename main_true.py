import numpy as np
from models import MLP, RegLog
from utils.activations import tanh, softmax
from utils.data_processing import one_hot_encoding, split_train_test
from utils.loss_functions import entropia_cruzada
from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
import argparse
import numpy as _np
import os as _os
import glob as _glob
from tqdm import tqdm

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

def normalize_features(X):
    X = X.astype(float)
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

def add_attributes(X, y, image_names, attr_file):
    print('loading attributes...')
    attr_df, attr_cols = _load_attributes(attr_file)

    A_list = []
    valid_idx = []

    for i, img in tqdm(enumerate(image_names)):
        if img in attr_df.index:
            A_list.append(attr_df.loc[img, attr_cols].values)
            valid_idx.append(i)

    if not A_list:
        raise RuntimeError('Nenhum atributo pôde ser alinhado com as imagens.')

    A = np.vstack(A_list)

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
    if args.features_dir:
        return load_celeba_dataset(args)
    else:
        return load_mock_dataset()

def build_model(args, num_features, num_classes):
    if args.model is None:
        raise ValueError("Use --model mlp ou reglog")

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

def evaluate_model(model, X_test, y_test):
    # X_test_bias = np.insert(X_test, 0, 1, axis=1)
    y_pred = model.predict(X_test)

    acc = np.mean(
        np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)
    )

    print(f"Test accuracy: {acc:.4f}")

def main(args):
    EPOCHS = 500

    # 1 - carregar dados
    X, y, num_features, num_classes = load_dataset(args)

    # 2 - split
    # X_train, y_train, X_test, y_test = stratified_split(X, y)
    X_train, y_train, X_test, y_test = split_train_test(X, y, rate=0.7)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test  = (X_test - mean) / std

    print('Train set shapes', X_train.shape, y_train.shape)
    print('Test set shapes', X_test.shape, y_test.shape)

    # 3 - modelo
    model = build_model(args, num_features, num_classes)

    # 4 - treino
    history_loss = model.fit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        learning_rate=1e-1,
        epochs=EPOCHS,
        optimizer=args.optimizer,
    )

    # 5 - avaliação
    evaluate_model(model, X_test, y_test)

    # 6 - plot
    plot_kfold_losses(history_loss)

if __name__ == "__main__":
    main(args)