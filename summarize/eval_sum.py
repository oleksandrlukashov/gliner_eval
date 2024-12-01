import torch
from sentence_transformers import SentenceTransformer
import json
from nltk.tokenize import word_tokenize
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

model = SentenceTransformer('all-MiniLM-L6-v2')

dataset = load_dataset('cnn_dailymail', '3.0.0')['validation']

with open('summarize/summarize_results_20241129_003735.json', 'r') as f:
    results = json.load(f)

def compute_cosine_similarity(reference_summaries, predicted_summaries):
    similarities = []
    for ref, pred in zip(reference_summaries, predicted_summaries):
        if ref and pred: 
            embeddings = model.encode([ref, pred])
            ref_tensor = torch.tensor(embeddings[0])
            pred_tensor = torch.tensor(embeddings[1])
            ref_tensor = ref_tensor / ref_tensor.norm(p=2)
            pred_tensor = pred_tensor / pred_tensor.norm(p=2)
            similarity = torch.nn.functional.cosine_similarity(ref_tensor.unsqueeze(0), pred_tensor.unsqueeze(0)).item()
            similarities.append(similarity)
    return np.mean(similarities) if similarities else 0.0

def compute_rouge(reference_summaries, predicted_summaries):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for ref, pred in zip(reference_summaries, predicted_summaries):
        if ref and pred:
            scores = scorer.score(ref, pred)
            for key in scores:
                rouge_scores[key].append(scores[key].fmeasure)

    return {key: np.mean(value) for key, value in rouge_scores.items()}

def compute_bleu(reference_summaries, predicted_summaries):
    bleu_scores = []
    for ref, pred in zip(reference_summaries, predicted_summaries):
        if ref and pred:
            reference = ref.split()
            candidate = pred.split()
            bleu_scores.append(sentence_bleu([reference], candidate))
    return np.mean(bleu_scores) if bleu_scores else 0.0


reference_summaries = []
predicted_summaries = []

for idx, entry in enumerate(dataset):
    str_idx = str(idx)
    if str_idx in results and 'summary' in results[str_idx] and 'score' in results[str_idx]:
        predicted_summary = results[str_idx].get('summary', None)
        score = results[str_idx].get('score', None)

        if predicted_summary is not None and score is not None:
            reference_summary = entry['highlights']
            predicted_summaries.append(predicted_summary)
            reference_summaries.append(reference_summary)

avg_similarity = compute_cosine_similarity(reference_summaries, predicted_summaries)
print(f"Avg Cosine Similarity: {avg_similarity:.4f}")

rouge_scores = compute_rouge(reference_summaries, predicted_summaries)
print("ROUGE Scores:")
for key, value in rouge_scores.items():
    print(f"{key}: {value:.4f}")

avg_bleu = compute_bleu(reference_summaries, predicted_summaries)
print(f"Mean BLEU: {avg_bleu:.4f}")
