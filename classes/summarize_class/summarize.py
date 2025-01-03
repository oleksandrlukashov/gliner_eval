import torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from nltk.tokenize import word_tokenize
import numpy as np
import json
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parent_class import AbstractEvaluator

class SumGLiNEREvaluator(AbstractEvaluator):

  def prepare_dataset(self, dataset: Dataset, *args, **kwargs):
      text = dataset['article']
      labels = ['summary']
      true_labels = dataset['highlights']
      return text, true_labels, labels
  
  def prepare_text(self, text):
    input_texts = []
    prompt = 'Extract highlight sentence from given text:\n'
    for texts in text:
      if texts != '':
        input_texts.append(prompt + texts)

    return input_texts
      
  def __call__(self, text, labels, threshold: float=0.5):


    predictions = []
    for texts in text:

      prediction = self.model.predict_entities(texts, labels, threshold=0.5)
      predictions.append(prediction)

    return predictions
  
  def process_predictions(self, predictions):

      preds_ = []
      for preds in predictions:
        if preds != []:
          for dicts in preds:
            predicted_summary = dicts.get('text', '')
            preds_.append(predicted_summary)
        else:
          preds_.append('')

      return preds_

  def compute_f_score(self, predicts, true_labels):
    
    similarities = []
    bleu_scores = []
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    for true_label, predict in zip(true_labels, predicts):
      if predict != '':
        embeddings = encoder.encode([true_label, predict])
        ref_tensor = torch.tensor(embeddings[0])
        pred_tensor = torch.tensor(embeddings[1])
        ref_tensor = ref_tensor / ref_tensor.norm(p=2)
        pred_tensor = pred_tensor / pred_tensor.norm(p=2)
        similarity = torch.nn.functional.cosine_similarity(ref_tensor.unsqueeze(0), pred_tensor.unsqueeze(0)).item()
        similarities.append(similarity)
        scores = scorer.score(true_label, predict)
        for key in scores:
            rouge_scores[key].append(scores[key].fmeasure)
        reference = true_label.split()
        candidate = predict.split()
        bleu_scores.append(sentence_bleu([reference], candidate))
    
    return np.mean(bleu_scores), {key: np.mean(value) for key, value in rouge_scores.items()}, np.mean(similarities)


  def evaluate(self, dataset_id, config_name, labels=None, *args, **kwargs):

    dataset = load_dataset(dataset_id, config_name)['validation']
    dataset = dataset[:50]
    text, true_labels, labels = self.prepare_dataset(dataset)
    input_text = self.prepare_text(text)
    predictions = self.__call__(input_text, labels, threshold=0.5)
    preds = self.process_predictions(predictions)
    return self.compute_f_score(preds, true_labels)

evaluator = SumGLiNEREvaluator('knowledgator/gliner-multitask-v1.0')
results = evaluator.evaluate('cnn_dailymail', '3.0.0')
output_file = 'evaluation_results_sum_1.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

evaluator = SumGLiNEREvaluator('knowledgator/gliner-multitask-large-v0.5')
results = evaluator.evaluate('cnn_dailymail', '3.0.0')
output_file = 'evaluation_results_sum_05.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)