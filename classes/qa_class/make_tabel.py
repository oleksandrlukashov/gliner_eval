import json
import os
import pandas as pd

files = [
    'classes/qa_class/GLiNER Multitask Llama.json',
    'classes/qa_class/GLiNER Multitask v0.5.json',
    'classes/qa_class/GLiNER Multitask v1.0.json'
]

metrics_data = []

for filename in files:
    with open(filename, 'r') as file:
        data = json.load(file)

    metrics_data.append({
        'Model': os.path.splitext(os.path.basename(filename))[0],
        'HasAns Exact': data['HasAns_exact'],
        'HasAns F1': data['HasAns_f1']
    })

df = pd.DataFrame(metrics_data)

markdown_table = df.to_markdown(index=False)

output_path = 'classes/qa_class/qa.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)
