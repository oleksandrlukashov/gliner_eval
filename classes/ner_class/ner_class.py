import json
import re
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_dataset
from parent_class import AbstractEvaluator

class NerGLiNEREvaluator(AbstractEvaluator):
  
    def compute_f_score(self, true_labels, predicts):

      true_set = set(true_labels)
      pred_set = set(predicts)

      tp = len(true_set.intersection(pred_set))
      fp = len(pred_set - true_set)
      fn = len(true_set - pred_set)

      precision = tp / (tp + fp) if tp + fp > 0 else 0
      recall = tp / (tp + fn) if tp + fn > 0 else 0
      f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

      return precision, recall, f1, tp, fp, fn
    
    def process_predictions(self, predictions):
      preds = []
      for predict in predictions:
          for pred_ in predict:
            label = pred_.get('label')
            entity = pred_.get('text')
            result = entity + ' <> ' + label
            preds.append(result)
      return preds

    def __call__(self, texts, labels, threshold=0.5):
      predictions = []
      list_for_label = []
      for text, label_list in tqdm(zip(texts, labels), total=len(texts), desc='Predicting'):
          for label in label_list:
              list_for_label = []
              list_for_label.append(label)
              prediction = self.model.predict_entities(text, list_for_label, threshold=0.5)
              predictions.append(prediction)

      return predictions

    def prepare_dataset(self, dataset, text_column='sentence', ner_column='ner', *args, **kwargs):

      senteces = dataset[text_column]
      entities = dataset[ner_column]
      grouped_labels = []
      true_labels = []
      texts_by_line = []

      for texts_lists, ent_lists in zip(senteces, entities):

            current_labels = []

            for ent_dicts in ent_lists:
                  
                  id_start = ent_dicts.get('id-start', None)
                  id_end = ent_dicts.get('id-end', None)
                  label = ent_dicts.get('entity-type', None)
                  current_labels.append(label)
                  true_label = ' '.join(texts_lists[id_start:id_end+1]) + ' <> ' + label
                  true_labels.append(true_label)

            text = ' '.join(texts_lists)
            text = re.sub(r'\s([?.!,;])', r'\1', text)

            texts_by_line.append(text)
            current_labels = list(set(current_labels))
            grouped_labels.append(current_labels)
            texts_by_line = [x for x in texts_by_line if x != []]
            grouped_labels = [x for x in grouped_labels if x != []]

      return texts_by_line, grouped_labels, true_labels
      
    def evaluate(self, dataset_id, config_name, labels=None, *args, **kwargs):
        dataset = load_dataset(dataset_id, config_name)['test']
        text, labels, true_labels = self.prepare_dataset(dataset)
        predictions = self.__call__(text, labels)
        preds = self.process_predictions(predictions)
        return self.compute_f_score(true_labels, preds)