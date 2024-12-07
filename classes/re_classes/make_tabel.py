import json
import os
import pandas as pd

files = [
    'classes/re_classes/docre_llama.json',
    'classes/re_classes/docre_05.json',
    'classes/re_classes/docre_1.json' ]

metrics_data = []

for filename in files:
    try:
        with open(filename, 'r') as file:
            data = json.load(file)

        precision = data[0]
        recall = data[1]
        f1_score = data[2]
        metrics_data.append({
            'File': os.path.basename(filename),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"error with {filename}: {e}")

df = pd.DataFrame(metrics_data)

markdown_table = df.to_markdown(index=False)

output_path = 'classes/re_classes/docre.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)
