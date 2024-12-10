import json
import pandas as pd
import os

files = [
    'classes/summarize_class/GLiNER Multitask Llama.json',
    'classes/summarize_class/GLiNER Multitask v0.5.json',
    'classes/summarize_class/GLiNER Multitask v1.0.json'
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
            'Model': os.path.splitext(os.path.basename(filename))[0],
            'BLEU': bleu,
            'ROUGE1': rouge1,
            'ROUGE2': rouge2,
            'ROUGEL': rougel,
            'Cosine Similarity': cosine_similarity
        })

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error with {filename}: {e}")
df = pd.DataFrame(metrics_data)
markdown_table = df.to_markdown(index=False)
output_path = 'classes/summarize_class/sum.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)

print(f"Markdown table saved to {output_path}")
