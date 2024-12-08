import json
import os
import pandas as pd

files = [
    'classes/qa_class/GLiNER Llama Multitask.json',
    'classes/qa_class/GLiNER Multitask v0.5.json',
    'classes/qa_class/GLiNER Multitask v1.0.json'
]

metrics_data = []

for filename in files:
    with open(filename, 'r') as file:
        data = json.load(file)

    precision = data['precision']
    recall = data['recall']
    f1_score = data['f1_score']

    metrics_data.append({
        'Model': os.path.splitext(os.path.basename(filename))[0],
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    })

df = pd.DataFrame(metrics_data)

markdown_table = df.to_markdown(index=False)

output_path = 'classes/qa_class/qa.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)

