import matplotlib.pyplot as plt
import json
import os

files = [
    'classes/qa_class/GLiNER Multitask Llama.json',
    'classes/qa_class/GLiNER Multitask v0.5.json',
    'classes/qa_class/GLiNER Multitask v1.0.json'
]

metrics_data = {
    'HasAns Exact': [],
    'HasAns F1': []
}

for filename in files:
    with open(filename, 'r') as file:
        data = json.load(file)

    metrics_data['HasAns Exact'].append(data['HasAns_exact'])
    metrics_data['HasAns F1'].append(data['HasAns_f1'])

file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]

x = range(len(file_names))

plt.bar(x, metrics_data['HasAns Exact'], width=0.3, label='HasAns Exact', align='center')
plt.bar([i + 0.3 for i in x], metrics_data['HasAns F1'], width=0.3, label='HasAns F1', align='center')

plt.title('Performance of GLiNER Models (HasAns Metrics)')
plt.ylabel('Values')
plt.xlabel('Models')

plt.xticks([i + 0.15 for i in x], file_names, rotation=45)

plt.legend()

plt.tight_layout()
plt.show()
