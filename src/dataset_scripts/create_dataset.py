import pandas as pd
import os

def processar_dataset_daigt():
    input_path = "../../data/raw/dataset_kaggle_daigt.csv"
    output_path = "../../data/processed/dataset_kaggle_daigt_processed.csv"

    print("A ler o dataset do Kaggle...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"ERRO: Não encontrei {input_path}")
        return

    mapeamento_marcas = {
        'persuade_corpus': 'Human',
        'train_essays': 'Human',
        'chat_gpt_moth': 'OpenAI',
        'radekgpt4': 'OpenAI',
        'radek_500': 'OpenAI',
        'kingki19_palm': 'Google',
        'palm-text-bison1': 'Google',
        'llama2_chat': 'Meta',
        'llama_70b_v1': 'Meta',
        'NousResearch/Llama-2-7b-chat-hf': 'Meta',
        'darragh_claude_v6': 'Anthropic',
        'darragh_claude_v7': 'Anthropic'
    }

    df['Label'] = df['source'].map(mapeamento_marcas)

    classes_oficiais = ['Human', 'OpenAI', 'Meta', 'Google', 'Anthropic']
    df_filtrado = df[df['Label'].isin(classes_oficiais)].copy()

    print("A cortar os textos para 120 carateres...")
    df_filtrado['text'] = df_filtrado['text'].astype(str).str.slice(0, 120)
    
    df_filtrado = df_filtrado.drop_duplicates(subset=['text'])

    min_amostras = df_filtrado['Label'].value_counts().min()
    df_equilibrado = df_filtrado.groupby('Label').sample(n=min_amostras, random_state=42)
    
    df_equilibrado['ID'] = [f"DAIGT-{i+1}" for i in range(len(df_equilibrado))]
    df_equilibrado.rename(columns={'text': 'Text'}, inplace=True)
    
    df_final = df_equilibrado[['ID', 'Text', 'Label']]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, sep=';', index=False)

    print("\nSUCESSO!")
    print(f"Ficheiro guardado em: {output_path}")
    print(f"Total de linhas: {len(df_final)}")
    print("\nDistribuição das classes:")
    print(df_final['Label'].value_counts())

if __name__ == "__main__":
    processar_dataset_daigt()