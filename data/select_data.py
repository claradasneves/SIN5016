import shutil
import os
import sys
from pathlib import Path
import pandas as pd

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
        caminho_destino = os.path.join(pasta_destino, img)

        if os.path.exists(caminho_origem):
            shutil.copy(caminho_origem, caminho_destino)
            print(f"  ✓ Copiado: {img}")
        else:
            print(f"  ✗ Arquivo não encontrado: {img}")


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
    if pasta_imagens is None:
        pasta_imagens = os.path.join(SCRIPT_DIR, "img_align_celeba/img_align_celeba")
    if destino is None:
        destino = os.path.join(SCRIPT_DIR, "selected_images/")
    if ids_procurados is None:
        ids_procurados = ["2937"] # Exemplo de ID padrão, aceita IDs múltiplos
    
    # Carregar dados
    df = carregar_dados(arquivo_txt)
    
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


if __name__ == "__main__":
    main()