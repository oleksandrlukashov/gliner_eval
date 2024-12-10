import matplotlib.pyplot as plt
import json
import os

files = [
    'classes/summarize_class/GLiNER Multitask Llama.json',
    'classes/summarize_class/GLiNER Multitask v0.5.json',
    'classes/summarize_class/GLiNER Multitask v1.0.json'
]

metrics_data = {
    'BLEU': [],
    'ROUGE1': [],
    'ROUGE2': [],
    'ROUGEL': [],
    'Cosine Similarity': []
}

for filename in files:
    with open(filename, 'r') as file:
        data = json.load(file)

    bleu = data[0]
    rouge1 = data[1]['rouge1']
    rouge2 = data[1]['rouge2']
    rougel = data[1]['rougeL']
    cosine_similarity = data[2]
    
    metrics_data['BLEU'].append(bleu)
    metrics_data['ROUGE1'].append(rouge1)
    metrics_data['ROUGE2'].append(rouge2)
    metrics_data['ROUGEL'].append(rougel)
    metrics_data['Cosine Similarity'].append(cosine_similarity)

file_names = [os.path.splitext(os.path.basename(file))[0] for file in files]

x = range(len(file_names))

width = 0.15

plt.bar(x, metrics_data['BLEU'], width=width, label='BLEU', align='center')
plt.bar([i + width for i in x], metrics_data['ROUGE1'], width=width, label='ROUGE1', align='center')
plt.bar([i + 2 * width for i in x], metrics_data['ROUGE2'], width=width, label='ROUGE2', align='center')
plt.bar([i + 3 * width for i in x], metrics_data['ROUGEL'], width=width, label='ROUGEL', align='center')
plt.bar([i + 4 * width for i in x], metrics_data['Cosine Similarity'], width=width, label='Cosine Similarity', align='center')

plt.title('Performance of GLiNER models on Summarization task')
plt.ylabel('Values')

plt.xticks([i + 2 * width for i in x], file_names, rotation=45)

plt.legend()

plt.tight_layout() 
plt.show()
