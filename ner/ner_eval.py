import json

def extract_label_text_pairs(data):

    label_text_pairs = {}

    for key, entities in data.items():
        for entity in entities:
            label = entity['label']
            text = entity['text']

            if label not in label_text_pairs:
                label_text_pairs[label] = []

            label_text_pairs[label].append(text)
    
    return label_text_pairs

def extract_predicted_words(data):

    results_dict = {}

    for key, entities in data.items():
        for entity in entities:
            label = entity['label']
            text = entity['text']
            words = text.split()

            if label not in results_dict:
                results_dict[label] = []

            results_dict[label].extend(words)

    return results_dict

def similarity(str1, str2):

    return str1.lower() == str2.lower()

def evaluate_metrics(dataset_dict, results_dict):

    tp = 0  
    fp = 0  
    fn = 0  
    
    for label, dataset_words in dataset_dict.items():
        if label not in results_dict:
            fn += len(dataset_words)
            continue
        
        result_words = results_dict[label]
        matched_results = set()

        for dataset_word in dataset_words:
          
            if any(
                similarity(dataset_word, result_word)
                for result_word in result_words
            ):
                tp += 1
                matched_results.add(dataset_word)
            else:
                fn += 1 

        for result_word in result_words:
            if result_word not in matched_results:
                fp += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, f1

file_path_1 = 'eval/ner/ner_results_20241128_152646.json'
with open(file_path_1, 'r') as f:
    data_1 = json.load(f)

file_path_2 = 'eval/ner/unie_synthetic.json'
with open(file_path_2, 'r') as f:
    data_2 = json.load(f)

results_dict = extract_predicted_words(data_1)

allowed_labels = ["Person", "Country", "Location", "City", "Event", "Organization", 'State']
dataset_dict = {}

for entry in data_2:
    if 'ner' in entry and 'tokenized_text' in entry:
        tokenized_text = entry['tokenized_text']
        ner_entities = entry['ner']
        for ner in ner_entities:
            start_idx, end_idx, label = ner
            if label in allowed_labels:
                words = tokenized_text[start_idx:end_idx + 1]
                if label not in dataset_dict:
                    dataset_dict[label] = []
                dataset_dict[label].extend(words)

precision, recall, f1 = evaluate_metrics(dataset_dict, results_dict)

print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1-score: {f1:.2%}")
