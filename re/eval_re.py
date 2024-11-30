import json
import re

def extract_triplets_from_preds(pred_file_path):
    triplets = []

    with open(pred_file_path, 'r', encoding='utf-8') as file:
        preds = json.load(file)

        for key, value in preds.items():

            match = re.match(r"^(.*?)\s*<>\s*(.*?)\s*<>\s*(.*)$", key)
            if match:
                head = match.group(1) 
                relation = match.group(2) 
                tail = match.group(3) 
                for item in value:
                    triplet = {
                        'head': head,
                        'relation': relation,
                        'tail': tail
                    }
                    triplets.append(triplet)

    return triplets


def extract_triplets_from_real_data(file_path):

    triplets = [] 

    with open(file_path, 'r') as file:
        for line in file:
            try:
                entry = json.loads(line.strip()) 
                names = entry.get('names', [])
                head = entry.get('head', {}).get('text', '')
                tail = entry.get('tail', {}).get('text', '')

                for name in names:
                    triplet = {
                        'head': head,
                        'relation': name,
                        'tail': tail
                    }
                    triplets.append(triplet)

            except json.JSONDecodeError:

                continue

    return triplets


file_path = 're/val_wiki-2.json' 
triplets = extract_triplets_from_real_data(file_path)
predicts = extract_triplets_from_preds('re/relation_extraction_results_20241129_203053.json')


def compare_triplets(pred_triplets, real_triplets):
    correct_count = 0

    for pred in pred_triplets:

        if any(pred['head'] == real['head'] and pred['tail'] == real['tail'] and pred['relation'] == real['relation'] for real in real_triplets):
            correct_count += 1
    
    return correct_count



correct_matches = compare_triplets(predicts, triplets)
print(f"Correct matches: {correct_matches}")
def calculate_metrics(pred_triplets, real_triplets):

    true_positives = 0
    for pred in pred_triplets:

        if any(pred['head'] == real['head'] and pred['tail'] == real['tail'] and pred['relation'] == real['relation'] for real in real_triplets):
            true_positives += 1

    false_positives = len(pred_triplets) - true_positives

    false_negatives = 0
    for real in real_triplets:
        if not any(real['head'] == pred['head'] and real['tail'] == pred['tail'] and real['relation'] == pred['relation'] for pred in pred_triplets):
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0

    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    exact_match = len(real_triplets) / len(pred_triplets) if len(pred_triplets) > 0 else 0

    return precision, recall, f1_score, exact_match

precision, recall, f1_score, exact_match = calculate_metrics(predicts, triplets)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")
print(f"Exact match accuracy (Exact Score): {exact_match:.4f}")
