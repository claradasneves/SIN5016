import os
import sys
import numpy as np
import glob

# Garantir imports relativos funcionem mesmo quando o script é executado de outro diretório
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))

from utils.data_processing import extract_hog_from_paths


def pipeline_extract_feature(selecionadas_dir: str = None, resize_to=(128, 128)):
	"""Pipeline simples: pega imagens em `selecionadas` e extrai HOG usando utilitário existente.

	Args:
		selecionadas_dir: Pasta contendo imagens (se None, usa `selecionadas` dentro de `data/`).
		resize_to: Tupla (H, W) para redimensionar antes da extração (ou None para não redimensionar).

	Returns:
		Numpy array com features HOG (N, D)
	"""
	if selecionadas_dir is None:
		selecionadas_dir = os.path.join(SCRIPT_DIR, 'selected_images')

	if not os.path.isdir(selecionadas_dir):
		raise FileNotFoundError(f"Pasta não encontrada: {selecionadas_dir}")

	# coletar imagens jpg/jpeg/png
	patterns = [os.path.join(selecionadas_dir, ext) for ext in ('*.jpg', '*.jpeg', '*.png')]
	paths = []
	for p in patterns:
		paths.extend(sorted(glob.glob(p)))

	if not paths:
		raise FileNotFoundError(f"Nenhuma imagem encontrada em: {selecionadas_dir}")

	print(f"Processando {len(paths)} imagens de: {selecionadas_dir}")

	X_hog = extract_hog_from_paths(paths, resize_to=resize_to)

	print(f"Features extraídas. shape = {X_hog.shape}")

	out_file = os.path.join(SCRIPT_DIR, 'hog_features.npy')
	np.save(out_file, X_hog)
	print(f"Features salvas em: {out_file}")

	return X_hog


if __name__ == '__main__':
	try:
		pipeline_extract_feature()
	except Exception as e:
		print(f"Erro: {e}")
		raise