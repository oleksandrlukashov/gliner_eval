import json
import collections
import re
import string

with open('eval/re/unie_synthetic.json', 'r', encoding='utf-8') as file:
    data_2 = json.load(file)

with open('eval/re/relation_extraction_results_20241129_040256.json', 'r', encoding='utf-8') as file:
    data_1 = json.load(file)

real_triples = []

for entry in data_2:
    if 'tokenized_text' in entry and 'ner' in entry:
        tokenized_text = entry['tokenized_text']
        ner_entities = entry['ner']

        entities = {}
        
        for start_idx, end_idx, label in ner_entities:
            entity = ' '.join(tokenized_text[start_idx:end_idx+1])
            entities[label] = entity
        
        if 'Person' in entities and 'Location' in entities:
            real_triples.append((entities['Person'], entities['Location'], 'relation'))
        if 'Person' in entities and 'Organization' in entities:
            real_triples.append((entities['Person'], entities['Organization'], 'relation'))
        if 'Person' in entities and 'Date' in entities:
            real_triples.append((entities['Person'], entities['Date'], 'relation'))
        if 'Person' in entities and 'Country' in entities:
            real_triples.append((entities['Person'], entities['Country'], 'relation'))
        if 'Person' in entities and 'City' in entities:
            real_triples.append((entities['Person'], entities['City'], 'relation'))


predicted_triples = []

for key, value in data_1.items():
    for pred in value:

        predicted_text = pred.get('text', '')
        label = pred.get('label', '')

        if "<entity1>" in label and "<entity2>" in label:

            entities = predicted_text.split()
            if len(entities) >= 2:
                entity1 = entities[0]
                entity2 = entities[-1]
                predicted_triples.append((entity1, entity2, 'relation'))

def normalize_answer(s):
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s: return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_raw_scores(predictions, actuals):
    exact_scores = []
    f1_scores = []
    for pred, actual in zip(predictions, actuals):
        exact_scores.append(compute_exact(actual, pred))
        f1_scores.append(compute_f1(actual, pred))
    return exact_scores, f1_scores

exact_scores, f1_scores = get_raw_scores(predicted_triples, real_triples)
out_eval = {
    'exact': 100.0 * sum(exact_scores) / len(exact_scores),
    'f1': 100.0 * sum(f1_scores) / len(f1_scores),
    'total': len(predicted_triples)
}

print(f"Exact Match: {out_eval['exact']}%")
print(f"F1 Score: {out_eval['f1']}%")
