# SIN5016
Repositório que contém os trabalhos parciais e finais da disciplina SIN5016 - Aprendizado de Máquina

Modelo para tarefa de classificação com múltiplas classes (com pelo menos 20% da base) (identificação biométrica)


# TO-DO

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
