import torch
from datasets import load_dataset
from gliner import GLiNER
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parent_class import AbstractEvaluator
import json
from tqdm import tqdm

class OpenIeGliNEREvaluator(AbstractEvaluator):

    def prepare_dataset(self, dataset, *args, **kwargs):

      true_labels=[]
      texts=[]
      for merge_key, merge_data in dataset.items():
        for item in merge_data:
            for relation in item.get("tuples", []):
              arg1 = relation.get("arg1", {}).get('text')
              rel = relation.get('rel',{}).get('text')
              arg2 = relation.get('arg2',{}).get('text')
              true_label = arg1 + ' ' + rel + ' ' + arg2
              true_labels.append(true_label)
            sentence = item.get('sent',[])
            texts.append(sentence)

      return texts, true_labels

    def compute_f_score(self, predicts, true_labels, texts):
      tp = 0 
      fp = 0  
      fn = 0  

      for predict in predicts:
          if any(predict in true for true in true_labels): 
              tp += 1
          else:
              fp += 1

      for true in true_labels:
          if not any(true in predict for predict in predicts):
              fn += 1

      precision = tp / (tp + fp) if (tp + fp) > 0 else 0
      recall = tp / (tp + fn) if (tp + fn) > 0 else 0
      f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
      print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
      return precision, recall, f1

    def process_predictions(self, predictions):
      preds = []
      for sublist in predictions:
        for dicts in sublist:
          pred = dicts.get('text')
          preds.append(pred)
      return preds



    def __call__(self, texts, labels, threshold=0.5):

      predictions = []
      for inputs in texts:
        prediction = self.model.predict_entities(inputs, labels, threshold=threshold)
        predictions.append(prediction)
      
      predictions = [sublist for sublist in predictions if sublist]

      return predictions

    def evaluate(self, dataset, labels=None, *args, **kwargs):

      test_texts, true_labels = self.prepare_dataset(dataset)
      predictions = self.__call__(test_texts, ['entities'], threshold=0.5)
      preds = self.process_predictions(predictions)
      return self.compute_f_score(preds,true_labels, test_texts)

with open('classes/open_class/WiRe57_343-manual-oie.json', 'r', encoding='utf-8') as file:
    dataset = json.load(file)
evaluator = OpenIeGliNEREvaluator('knowledgator/gliner-multitask-v1.0')
results = evaluator.evaluate(dataset)
output_file = 'evaluation_results_oie_1.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

evaluator = OpenIeGliNEREvaluator('knowledgator/gliner-multitask-large-v0.5')
results = evaluator.evaluate(dataset)
output_file = 'evaluation_results_oie_05.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)