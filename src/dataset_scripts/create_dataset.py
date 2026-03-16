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
        # HUMANOS
        'persuade_corpus': 'Human',
        'train_essays': 'Human',
        
        # OPENAI
        'chat_gpt_moth': 'OpenAI',
        'radekgpt4': 'OpenAI',
        'radek_500': 'OpenAI',
        
        # GOOGLE
        'kingki19_palm': 'Google',
        'palm-text-bison1': 'Google',
        
        # META
        'llama2_chat': 'Meta',
        'llama_70b_v1': 'Meta',
        'NousResearch/Llama-2-7b-chat-hf': 'Meta',
        
        # MISTRAL
        'mistral7binstruct_v2': 'Mistral',
        'mistral7binstruct_v1': 'Mistral',
        'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral',

        # ANTHROPIC
        'darragh_claude_v6': 'Anthropic',
        'darragh_claude_v7': 'Anthropic'
    }

    df['Label'] = df['source'].map(mapeamento_marcas)

    classes_oficiais = ['Human', 'OpenAI', 'Meta', 'Mistral', 'Google', 'Anthropic']
    df_filtrado = df[df['Label'].isin(classes_oficiais)].copy()

    df_equilibrado = df_filtrado.groupby('Label').sample(n=1000, random_state=42, replace=True).drop_duplicates()
    df_equilibrado['ID'] = [f"DAIGT-{i+1}" for i in range(len(df_equilibrado))]
    df_equilibrado.rename(columns={'text': 'Text'}, inplace=True)
    
    df_final = df_equilibrado[['ID', 'Text', 'Label']]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, sep=';', index=False)

    print("\nSUCESSO TOTAL!")
    print(f"Ficheiro guardado em: {output_path}")
    print(f"Total de linhas (Textos Reais): {len(df_final)}")
    print("\nDistribuição perfeita das tuas classes:")
    print(df_final['Label'].value_counts())

if __name__ == "__main__":
    processar_dataset_daigt()