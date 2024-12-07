import matplotlib.pyplot as plt
import json
import os

files = [
    'classes/qa_class/qa_llama.json',
    'classes/qa_class/qa_05.json',
    'classes/qa_class/qa_1.json'
]

metrics_data = {
    'Precision': [],
    'Recall': [],
    'F1': []
}

for filename in files:
    with open(filename, 'r') as file:
        data = json.load(file)

    metrics_data['Precision'].append(data['precision'])
    metrics_data['Recall'].append(data['recall'])
    metrics_data['F1'].append(data['f1_score'])

file_names = [os.path.basename(file) for file in files]

x = range(len(file_names))

plt.bar(x, metrics_data['Precision'], width=0.2, label='Precision', align='center')
plt.bar([i + 0.2 for i in x], metrics_data['Recall'], width=0.2, label='Recall', align='center')
plt.bar([i + 0.4 for i in x], metrics_data['F1'], width=0.2, label='F1', align='center')
plt.ylabel('Values')

plt.xticks([i + 0.2 for i in x], file_names, rotation=45)

plt.legend()

plt.tight_layout()
plt.show()