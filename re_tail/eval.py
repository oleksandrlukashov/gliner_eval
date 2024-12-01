import json

with open('ground_truth_.json', 'r') as f:
    ground_truth = json.load(f)

with open('relation_extraction_results_20241201_024635.json', 'r') as f:
    predictions = json.load(f)

ground_truth_strings = []
for key, value in ground_truth.items():
    for entry in value:
        head_relation = entry['head_relation']
        tail = entry['tail']

        ground_truth_strings.append(f"{head_relation} <> {tail}")

prediction_strings = []
for key, value in predictions.items():

    prediction_strings.append(key)

ground_truth_set = set(ground_truth_strings)
prediction_set = set(prediction_strings)

intersection = ground_truth_set.intersection(prediction_set)

TP = len(intersection)

FP = len(prediction_set - ground_truth_set)

FN = len(ground_truth_set - prediction_set)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

num_correct = TP  
num_total = len(prediction_strings) 

accuracy = num_correct / num_total if num_total > 0 else 0

print(f"correct ans: {num_correct}")
print(f"total : {num_total}")
print(f"precision: {precision:.4f}")
print(f"recall: {recall:.4f}")
print(f"f1-score: {f1_score:.4f}")
