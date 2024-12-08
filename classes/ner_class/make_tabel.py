import json
import os
import pandas as pd

files = [
    'classes/ner_class/GLiNER Llama Multitask.json',
    'classes/ner_class/GLiNER Multitask v0.5.json',
    'classes/ner_class/GLiNER Multitask v1.0.json'
]

metrics_data = []

for filename in files:
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        precision = data[0]
        recall = data[1]
        f1_score = data[2]
        metrics_data.append({
            'Model': os.path.splitext(os.path.basename(filename))[0],
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error with {filename}: {e}")

df = pd.DataFrame(metrics_data)
markdown_table = df.to_markdown(index=False)
output_path = 'classes/ner_class/ner.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)

print(f"Markdown table saved to {output_path}")
