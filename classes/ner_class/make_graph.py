import matplotlib.pyplot as plt
import json
import os

files = [
    'classes/ner_class/ner_llama.json',
    'classes/ner_class/ner_05.json',
    'classes/ner_class/ner_01.json'
]

metrics_data = {
    'Precision': [],
    'Recall': [],
    'F1': []
}

for filename in files:
    with open(filename, 'r') as file:
        data = json.load(file)

    metrics_data['Precision'].append(data[0])
    metrics_data['Recall'].append(data[1])
    metrics_data['F1'].append(data[2])

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