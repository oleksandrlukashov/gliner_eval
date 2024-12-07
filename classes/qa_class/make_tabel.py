import json
import os
import pandas as pd

files = [
    'classes/qa_class/qa_llama.json',
    'classes/qa_class/qa_05.json',
    'classes/qa_class/qa_1.json'
]

metrics_data = []

for filename in files:
    try:
        with open(filename, 'r') as file:
            data = json.load(file)

        precision = data.get('precision', None)
        recall = data.get('recall', None)
        f1_score = data.get('f1_score', None)

        if precision is not None and recall is not None and f1_score is not None:

            metrics_data.append({
                'File': os.path.basename(filename),
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1_score
            })
        else:
            print(f"Пропущены метрики в файле {filename}")

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Ошибка при обработке файла {filename}: {e}")

df = pd.DataFrame(metrics_data)

markdown_table = df.to_markdown(index=False)

output_path = 'classes/qa_class/qa.md'
with open(output_path, 'w') as f:
    f.write(markdown_table)

print(f"Markdown таблица сохранена в {output_path}")
