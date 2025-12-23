import shutil
import os
import sys
import numpy as np
import pandas as pd
import zipfile
import imageio
from hog import extract_hog_feature
import argparse


# Obter o diretório do script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Adicionar o diretório pai ao path para importar módulos locais
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, '..')))


def carregar_dados(arquivo_txt: str) -> pd.DataFrame:
    """
    Carrega o arquivo de identidades do CelebA.
    
    Args:
        arquivo_txt: Caminho para o arquivo identity_CelebA.txt
        
    Returns:
        DataFrame com colunas 'imagem' e 'id'
    """
    return pd.read_csv(arquivo_txt, sep=' ', header=None, names=['imagem', 'id'])


def obter_imagens_por_id(df: pd.DataFrame, id_procurado: str) -> list:
    """
    Obtém lista de imagens para um ID específico.
    
    Args:
        df: DataFrame com os dados das imagens
        id_procurado: ID da pessoa a buscar
        
    Returns:
        Lista de nomes de imagens para o ID
    """
    return df[df['id'] == int(id_procurado)]['imagem'].tolist()


def copiar_imagens(imagens: list, pasta_origem: str, pasta_destino: str, id_procurado: str) -> None:
    """
    Copia imagens de uma pasta para outra.
    
    Args:
        imagens: Lista de nomes de imagens a copiar
        pasta_origem: Caminho da pasta de origem
        pasta_destino: Caminho da pasta de destino
        id_procurado: ID sendo processado (para logging)
    """
    print(f"\nProcessando ID: {id_procurado} - {len(imagens)} imagens encontradas")
    
    for img in imagens:
        caminho_origem = os.path.join(pasta_origem, img)
        pasta_id = os.path.join(pasta_destino, str(id_procurado))
        os.makedirs(pasta_id, exist_ok=True)
        caminho_destino = os.path.join(pasta_id, img)

        if os.path.exists(caminho_origem):
            shutil.copy(caminho_origem, caminho_destino)
            print(f"  ✓ Copiado: {img}")
        else:
            print(f"  ✗ Arquivo não encontrado: {img}")

def obter_top_ids(df: pd.DataFrame, top_n: int = 2000) -> list:
    """
    Retorna os IDs mais frequentes no DataFrame.

    Args:
        df: DataFrame com colunas ['imagem', 'id']
        top_n: número de classes mais frequentes

    Returns:
        Lista de IDs (int) ordenados por frequência decrescente
    """
    contagem = df['id'].value_counts()
    top_ids = contagem.head(top_n).index.tolist()
    return top_ids


def main(arquivo_txt: str = None,
         pasta_imagens: str = None,
         ids_procurados: list = None,
         destino: str = None) -> None:
    """
    Função principal para selecionar e copiar imagens de múltiplas identidades.
    
    Args:
        arquivo_txt: Caminho para o arquivo de identidades
        pasta_imagens: Caminho para a pasta contendo as imagens
        ids_procurados: Lista de IDs a processar
        destino: Caminho para a pasta de destino
    """
    # Usar valores padrão relativos ao diretório do script

    if arquivo_txt is None:
        arquivo_txt = os.path.join(SCRIPT_DIR, "atributos/identity_CelebA.txt")
    
    # Carregar dados
    df = carregar_dados(arquivo_txt)
    
    if pasta_imagens is None:
        pasta_imagens = os.path.abspath("/Users/claradasneves/Downloads/archive/img_align_celeba/img_align_celeba")
    if destino is None:
        destino = os.path.join(SCRIPT_DIR, "selected_images/")
    if ids_procurados is None:
        ids_procurados = obter_top_ids(df, top_n=2000)
    
    # Criar pasta de destino
    os.makedirs(destino, exist_ok=True)
    print(f"Pasta de destino criada/verificada: {destino}")
    
    # Processar cada ID
    for id_procurado in ids_procurados:
        imagens = obter_imagens_por_id(df, id_procurado)
        if imagens:
            copiar_imagens(imagens, pasta_imagens, destino, id_procurado)
        else:
            print(f"\nAviso: Nenhuma imagem encontrada para o ID: {id_procurado}")
    
    print("\n✓ Processo concluído!")

def pipeline_celeba_hog(
    identity_txt: str,
    imagens_dir: str,
    out_dir: str,
    top_n: int = 2000,
):
    os.makedirs(out_dir, exist_ok=True)

    df = carregar_dados(identity_txt)
    top_ids = obter_top_ids(df, top_n)

    print(f"Processando {len(top_ids)} identidades")

    total = 0

    for person_id in top_ids:
        imagens = df[df['id'] == person_id]['imagem'].tolist()

        for img_name in imagens:
            img_path = os.path.join(imagens_dir, img_name)

            if not os.path.exists(img_path):
                print(f"Arquivo não encontrado: {img_name}")
                continue

            try:
                img = imageio.imread(img_path)

                hog_feat = extract_hog_feature(img)

                img_id = os.path.splitext(img_name)[0]
                out_path = os.path.join(out_dir, f"{img_id}.npy")

                np.save(out_path, hog_feat)

                total += 1
                if total % 1000 == 0:
                    print(f"  {total} imagens processadas")

            except Exception as e:
                print(f"Erro em {img_name}: {e}")

    print("\n✓ Pipeline finalizado")
    print(f"✓ Total de imagens processadas: {total}")
    print(f"✓ Features em: {out_dir}")

# Execução
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de extração de HOG do CelebA"
    )

    parser.add_argument(
        "--identity_txt",
        type=str,
        default=os.path.join(SCRIPT_DIR, "atributos/identity_CelebA.txt"),
        help="Caminho para o arquivo identity_CelebA.txt"
    )

    parser.add_argument(
        "--imagens_dir",
        type=str,
        required=True,
        help="Diretório contendo as imagens do CelebA"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "hog_npy"),
        help="Diretório de saída para os arquivos .npy"
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=2000,
        help="Número de identidades mais frequentes"
    )

    args = parser.parse_args()

    pipeline_celeba_hog(
        identity_txt=args.identity_txt,
        imagens_dir=args.imagens_dir,
        out_dir=args.out_dir,
        top_n=args.top_n,
    )
