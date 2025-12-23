# SIN5016
Repositório que contém os trabalhos parciais e finais da disciplina SIN5016 - Aprendizado de Máquina


## Como disparar o shell para experimentos?
No terminal execute:
>> chmod +x ./experiments.sh

A seguir execute:
>> ./experiments.sh

## Como rodar o código?
Para rodar pela primeira vez e gerar as imagens e extrair as features com HOG, rode primeiro o arquivo select_data.py
>> `python ./data/select_data.py`

No terminal execute: 
>> `python main.py --model mlp` para rodar a MLP

OU execute:
>>`python main.py --model reglog` para a Regressão Logística

Para adicionar Regularização, exemplo:
>> `python main.py --model <MODELO> --regularization l1`

### Opções adicionais
Você pode configurar a pasta de atributos, o arquivo de identidades e o número de classes utilizando os seguintes argumentos:

--features-dir
Caminho para a pasta contendo as features (ex.: HOG) ou para um arquivo .npy com todas as features.
>> `python main.py --model mlp --features-dir caminho/para/features`

--identity-file
Arquivo .txt que faz o mapeamento entre imagem e identidade (pessoa).
>> `python main.py --model reglog --identity-file caminho/para/identity.txt`

--num-classes
Número de classes (pessoas) a serem utilizadas. Por padrão, são selecionadas as classes com mais amostras.
>> `python main.py --model mlp --num-classes 300`

--use-attributes
Ativa o uso dos atributos semânticos do CelebA como features adicionais ao modelo.
Quando habilitado, os atributos binários (ex.: Smiling, Eyeglasses, Male, etc.) são concatenados às features extraídas por HOG, formando um vetor de características mais rico.
Esse argumento é opcional. Caso não seja especificado, o modelo utiliza apenas as features HOG extraídas das imagens.
>> `python main.py --model mlp --num-classes 50 --use-atributes`

--attr-file
Especifica o caminho para o arquivo CSV contendo os atributos do dataset CelebA.
Esse arquivo é utilizado somente se o argumento --use-attributes estiver ativado.
Por padrão, o caminho aponta para o arquivo oficial de atributos do CelebA previamente organizado no diretório do projeto.
>> `python main.py --model mlp --num-classes 50 --use-attributes --attr-file data/atributos/list_attr_celeba.csv`

## EXEMPLO 
>> `python main_true.py --features-dir {abs_path_features_hog} --model mlp -r l2 --num-classes 50 --use-attributes --attr-file {abs_path_attr}`

## Código
- Baixar o dataset
- Aplicar HOG (gera um histograma de orientações de cada pixel) nos dados (podemos escolher outro descritor, como  LPB (Local Pattern Binary) ou Transformada Wavelet) -> no caso de usar mais descritores, gerar outro modelo

- Escolher os dois tipos de classificadores (Modelos Lineares, Rede MLP, SVM,
Ensemble de Modelos Heterogêneo)
    - A arquitetura que deve ser adotada, no caso da MLP feedforward, consiste de
uma rede com 1 camada escondida, treinada com algoritmo de aprendizado
backpropagation, já no caso do SVM, deve ser o SVM tradicional (C-SVC).
- Validação com k-fold (k = 5)

- Principais funcionalidades que o professor precisa detecter no código (colocar como comentários)
    - parâmetros de configuração,
    - implementações de funções de ativação e de erro,
    - método de inicialização dos pesos,
    - algoritmo de aprendizado e
    - critérios de parada

## Relatório
Requisitos para o relatório.
- R01 Apresentar quais são e como se configura os parâmetros (incluindo tipo de descritor). 
- R02 Apresentar estruturas de dados que organizam os pesos que compõem as camadas da rede.
- R03 Apresentar como se deu a extração de características das instâncias no conjunto de dados.
- R04 Apresentar a estratégia de seleção de modelos (5-fold cross validation, por exemplo).
- R05 Apresentar como atuam os algoritmos de inicialização de pesos e a implementação do algoritmo de treinamento da rede.
- R06 Apresentar os resultados obtidos em forma de tabelas e gráficos.
- R07 Analisar os resultados obtidos
- R08 O relatório deverá ser elaborado seguindo o formato IEEE, disponível neste link, opção ’Template and Instructions on How to Create Your Paper’. As seções sugeridas não precisam ser seguidas: a ideia é usar a mesma diagramação, tamanho e tipo de fonte,
estilo dos parágrafos, margens, referências bibliográficas, etc. O arquivo deve ser convertido no formato PDF antes da submissão da entrega.
- R09 O relatório deve apresentar aspectos dos modelo. No caso de uma rede neural, a arquitetura selecionada e descrever seus parâmetros (como número de entradas, número de neurônios em cada camada, tipo de função de ativação de cada camada, função de
custo (ou de erro) aplicada na saída da rede, método de inicialização dos pesos, passo de aprendizado e critérios de parada).
- R10 O relatório deve apresentar o método utilizado para seleção dos valores adotados para os parâmetros, se houver inspiração em outros trabalhos publicados, cite-os adequadamente.
- R11 O relatório deve apresentar as curvas com a evolução dos erros de treinamento, de validação e de testes por época, conforme apresentado em sala.
- R12 O relatório deve ainda apresentar a acurácia média por classe (caractere), obtida nos testes para um ou dois descritores (modelo treinado com descritores HOG e o modelo treinado com descritor selecionado pelo grupo).
- R13 O relatório deve apresentar uma análise comparativa dos resultados obtidos pelos modelos induzidos a partir de diferentes descritores, justificando a diferença em termos de influência dos parâmetros.
- R14 O relatório deve apresentar uma análise comparativa dos resultados obtidos pelos modelos com melhor e pior desempenho, justificando em termos de influência dos parâmetros.

## Vídeo
- R01 Duração de 10 a 15 min, formato MP4, resolução suficiente para o código estar legível.
- R02 Cada membro deve gravar um exemplo explicando a codificação realizada por 02 modelos
- R03 Cada membro deve demonstrar conhecimento de todos os códigos desenvolvido
