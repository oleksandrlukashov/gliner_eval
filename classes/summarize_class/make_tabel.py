import json
import pandas as pd
import os

files = [
    'classes/summarize_class/sum_llama.json',
    'classes/summarize_class/sum_05.json',
    'classes/summarize_class/sum_1.json'
]
metrics_data = []
for filename in files:
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        bleu = data[0]
        rouge1 = data[1]['rouge1']
        rouge2 = data[1]['rouge2']
        rougel = data[1]['rougeL']
        cosine_similarity = data[2]

        metrics_data.append({
            'File': os.path.basename(filename),
            'BLEU': bleu,
            'ROUGE1': rouge1,
            'ROUGE2': rouge2,
            'ROUGEL': rougel,
            'Cosine Similarity': cosine_similarity
        })

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка при обработке файла {filename}: {e}")
df = pd.DataFrame(metrics_data)
markdown_table = df.to_markdown(index=False)
output_path = 'classes/summarize_class/sum.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)

