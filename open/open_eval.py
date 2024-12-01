import json
from datasets import load_dataset

dataset_sment_open = load_dataset("stanfordnlp/imdb")['test']

with open('open/open_extraction_results_20241129_045513.json', 'r', encoding='utf-8') as file:
    predicted_positive_parts = json.load(file)

def evaluate_predictions(reviews, predictions):
    total_reviews = len(reviews)
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, review in enumerate(reviews):
        pred_parts = predictions.get(str(i), [])

        for pred_part in pred_parts:
            if pred_part in review and dataset_sment_open[i]['label'] == 1:
                true_positives += 1
            elif pred_part not in review:
                false_negatives += 1
            else:
                false_positives += 1
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

reviews = dataset_sment_open['text']

precision, recall, f1 = evaluate_predictions(reviews, predicted_positive_parts)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
