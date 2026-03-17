# Projeto AP

## Constituição do Grupo

| Nome                             | Número  |
| -------------------------------- | ------- |
| João Manuel Machado da Cunha     | PG60272 |
| Lara Regina da Silva Pereira     | PG57884 |
| Mário Rafael Figueiredo da Silva | PG60282 |
| Pedro Manuel Dias Teixeira       | PG60294 |
| Fernando Jorge Silva Pires       | PG60253 |

## Estrutura do Repositório

```
Projeto-AP/
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── notebooks/                     # Notebooks Jupyter (experimentação e treino)
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_treino_numpy_dnn.ipynb
│   ├── 03_treino_cnn1d.ipynb
│   ├── 04_treino_bilstm.ipynb
│   ├── 05_treino_numpy_logisticregression.ipynb
│   ├── 06_submissao1_DNN.ipynb             # Submissao 1
│   └── 07_submissao1_cnn1d.ipynb           # Submissao 1
│
├── src/                           # Código-fonte
│   ├── data_processing.py         #   Pré-processamento de texto
│   ├── bag_of_words.py            #   Implementação Bag-of-Words (NumPy)
│   ├── features.py                #   Vocabulário, sequências e embeddings (PyTorch)
│   ├── hyperopt.py                #   Utilitários para grid search
│   ├── dataset_scripts/
│   │   └── create_dataset.py
│   ├── models_numpy/              #   Modelos implementados em NumPy
│   │   ├── dnn/                   #     DNN (camadas, ativações, loss, optimizador)
│   │   │   ├── neuralnet.py       #       Classe principal da rede neuronal
│   │   │   ├── layers.py          #       Camadas densas
│   │   │   ├── activation.py      #       Funções de ativação (ReLU, Softmax)
│   │   │   ├── losses.py          #       Funções de perda (Cross-Entropy)
│   │   │   ├── optimizer.py       #       Optimizador (SGD com momentum)
│   │   │   └── metrics.py         #       Métricas (accuracy)
│   │   └── logistic_regression.py #     Regressão logística
│   └── models_pytorch/            #   Modelos implementados em PyTorch
│       ├── cnn1d.py               #     CNN 1D para classificação de texto
│       └── bilstm.py              #     BiLSTM para classificação de texto
│
├── saved_models/                  # Modelos treinados (estão no .gitignore)
│
├── submissions/                   # Ficheiros de submissão
│   ├── submissao1/                #   Primeira submissão
│
│
├── docs/                          # Documentação adicional (#TODO)
├── requirements.txt               # Dependências Python
└── LICENSE
```
