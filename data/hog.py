import os
import sys
import numpy as np
import glob
import zipfile

import imageio.v2 as imageio
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize

# Garantir imports relativos funcionem mesmo quando o script é executado de outro diretório
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

def pipeline_extract_feature(
    selecionadas_dir: str = None,
    out_dir: str = None,
    resize_to=(64, 64),
):
    """
    Extrai HOG de cada imagem e salva um .npy por imagem,
    usando apenas o ID da imagem como nome do arquivo.
    """

    if selecionadas_dir is None:
        selecionadas_dir = os.path.join(SCRIPT_DIR, 'selected_images')

    if out_dir is None:
        out_dir = os.path.join(SCRIPT_DIR, 'hog_npy')

    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(selecionadas_dir):
        raise FileNotFoundError(f"Pasta não encontrada: {selecionadas_dir}")

    # buscar imagens dentro das subpastas (IDs das pessoas)
    patterns = [
        os.path.join(selecionadas_dir, '*', ext)
        for ext in ('*.jpg', '*.jpeg', '*.png')
    ]

    paths = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))

    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {selecionadas_dir}")

    print(f"Processando {len(paths)} imagens...")

    for i, img_path in enumerate(paths):
        try:
            # nome do arquivo sem extensão (ID da imagem)
            img_id = os.path.splitext(os.path.basename(img_path))[0]

            img = imageio.imread(img_path)

            hog_feat = extract_hog_feature(
                img,
                resize_to=resize_to,
            )

            out_path = os.path.join(out_dir, f"{img_id}.npy")
            np.save(out_path, hog_feat)

            if i % 1000 == 0:
                print(f"  {i}/{len(paths)} imagens processadas")

        except Exception as e:
            print(f"Erro na imagem {img_path}: {e}")

    # criar zip com todos os npy
    zip_path = out_dir + ".zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in os.listdir(out_dir):
            if file.endswith(".npy"):
                zipf.write(
                    os.path.join(out_dir, file),
                    arcname=file
                )

    print(f"\n✓ Extração finalizada")
    print(f"✓ Arquivos .npy em: {out_dir}")
    print(f"✓ ZIP criado em: {zip_path}")



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
            visualize=False,
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
            visualize=False,
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


if __name__ == '__main__':
	try:
		pipeline_extract_feature()
	except Exception as e:
		print(f"Erro: {e}")
		raise