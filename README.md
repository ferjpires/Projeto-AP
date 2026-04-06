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
│   ├── embeddings/
│   │   └── glove.6B.100d.txt      # Embeddings pré-treinados (ignorados pelo .gitignore por causa do tamanho)
│   ├── raw/
│   │   └── generated/             # Dados gerados de múltiplas fontes
│   │       ├── anthropic.csv
│   │       ├── google.csv
│   │       ├── human.csv
│   │       ├── meta.csv
│   │       └── openai.csv
│   ├── processed/
│   │   └── dataset_combined.csv   # Dataset combinado usado durante o treino dos modelos
│   └── validation/
│       ├── dataset-exemplos.csv
│       ├── subm1_labels_revealed.csv
│       └── subm2_labels_revealed.csv
├── notebooks/                     # Experiências e treinos documentados (nomes autoexplicativos)
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_treino_numpy_dnn.ipynb
│   ├── 03_treino_cnn1d.ipynb
│   ├── 04_treino_lstm.ipynb
│   ├── 05_treino_numpy_logisticregression.ipynb
│   ├── 06_treino_llm_few&zero_shot.ipynb
│   ├── 07_treino_distilbert.ipynb
│   └── 08_ensemble.ipynb
├── src/                           # Código de pré-processamento, vetorização e modelos
│   ├── data_processing.py         # Preparação geral do texto
│   ├── features.py                # Transformações utilizadas nos modelos
│   ├── hyperopt.py                # Otimização de hiperparâmetros
│   ├── stylometric_features.py    # Características estilométricas
│   ├── vectorizer.py
│   ├── tfidf.py
│   ├── models_numpy/
│   │   ├── logistic_regression.py # Regressão logística
│   │   └── dnn/                   # Implementação DNN raiz
│   │       ├── activation.py
│   │       ├── layers.py
│   │       ├── losses.py
│   │       ├── metrics.py
│   │       ├── neuralnet.py
│   │       └── optimizer.py
│   ├── models_pytorch/            # Modelos PyTorch
│   │   ├── cnn1d.py
│   │   ├── distilbert.py
│   │   └── lstm.py
│   ├── models_llm/                # Integração com llms
│   │   ├── __init__.py
│   │   ├── llm_client.py
│   │   └── llm_utils.py
│   └── dataset_scripts/           # Scripts para recolha e junção de dados
│       ├── fetch_human_wikipedia.py
│       ├── generate_google.py
│       ├── generate_meta.py
│       ├── generate_combined_dataset.py
│       └── utils.py
├── saved_models/                  # Pesos e vetorizadores guardados
│   ├── cnn1d_final_model.pt
│   ├── cnn1d_final_config.json
│   ├── cnn1d_final_style_extractor.pkl
│   ├── cnn1d_final_style_stats.npz
│   ├── cnn1d_final_vocab.pkl
│   ├── dnn_final_model.npz
│   ├── dnn_final_vectorizer.pkl
│   └── ensemble_notebook_probs.npz
├── Subm1/                         # Resultados da submissão 1
│   ├── subm1-g8-MEI-A.ipynb
│   ├── subm1-g8-MEI-B.ipynb
│   ├── subm1-g8-MEI-A.csv
│   ├── subm1-g8-MEI-B.csv
│   └── validation_data/subm1.csv
├── Subm2/                         # Resultados da submissão 2
│   ├── subm2-g8-MEI-A.ipynb
│   ├── subm2-g8-MEI-B.ipynb
│   ├── subm2-g8-MEI-A.csv
│   ├── subm2-g8-MEI-B.csv
│   └── validation_data/subm2.csv
├── Subm3/                         # Resultados da submissão 3
│   ├── subm3-g8-MEI-A.ipynb
│   ├── subm3-g8-MEI-B.ipynb
│   ├── subm3-g8-MEI-A.csv
│   ├── subm3-g8-MEI-B.csv
│   └── validation_data/subm3.csv
├── docs/                          # Documentação e enunciados
│   ├── prompts.md
│   └── enunciadoTrabalho.pdf
├── Presentation/
│   └── apresentacao.mp4           # Vídeo da apresentação
├── requirements.txt
└── LICENSE
```
