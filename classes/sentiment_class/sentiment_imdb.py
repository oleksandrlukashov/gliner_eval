import torch
from datasets import load_dataset
from gliner import GLiNER
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from abc import ABC, abstractmethod
from classes.parent_class import AbstractEvaluator
import re
from tqdm import tqdm

class ImbdGliNEREvaluator(AbstractEvaluator):

    def prepare_dataset(self, dataset, *args, **kwargs):
        edited_true = []
        true_labels = dataset['label']
        for label in true_labels:
            if label == 0:
              edited_true.append('negative')
            else:
              edited_true.append('positive')

        texts = dataset['text']
        labels = ['positive', 'negative']

        return texts, edited_true, labels


    def compute_f_score(self, predicts, true_labels):

        tp = 0  
        fp = 0 
        fn = 0

        for true, predict in zip(true_labels, predicts):
          if true == predict:
              tp += 1 
          else:
              if predict == 'positive':
                  fp += 1
              else:
                  fn += 1

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        
        return precision, recall, f1
          


    def process_predictions(self, predictions):
        final_labels = []
        for idx, prediction in predictions.items():
            sentiment_scores = defaultdict(float)
            for sentiment in prediction.get("sentiment", []):
                sentiment_scores[sentiment["label"]] += sentiment["score"]
            if sentiment_scores:
                best_label = max(sentiment_scores, key=sentiment_scores.get)
                final_labels.append(best_label)
            else:
                final_labels.append(None)

        return final_labels

    def __call__(self, texts, labels, threshold=0.5):

        predictions = {}

        for idx, input_text in enumerate(texts):
          if not input_text.strip():
            predictions[idx] = {"text": input_text, "sentiment": []}
            continue
          try:
            sentiment_result = self.model.predict_entities(
                      input_text, labels=labels, flat_ner=False, threshold=threshold
                  )
            predictions[idx] = {
                      "text": input_text,
                      "sentiment": [
                          {"label": entity["label"], "score": entity["score"]}
                          for entity in sentiment_result
                      ],
                  }
          except Exception as e:
              predictions[idx] = {"text": input_text, "sentiment": []}
          
        return predictions

    def evaluate(self, dataset_id, labels=None, *args, **kwargs):

      dataset = load_dataset(dataset_id)['test']
      test_texts, true_labels, labels = self.prepare_dataset(dataset)
      predictions = self.__call__(test_texts, labels, threshold=0.5)
      preds = self.process_predictions(predictions)
      return self.compute_f_score(preds, true_labels)