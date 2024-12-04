from datasets import load_dataset
from tqdm import tqdm
from gliner import GLiNER
import json
from classes.parent_class import AbstractEvaluator

class DocredReGLiNEREvaluator(AbstractEvaluator):

    def prepare_dataset(self, json_data, rel_info, text_column='sents', rel_column='labels', *args, **kwargs):
        grouped_labels = []
        true_labels = []
        texts_by_line = []

        for item in json_data:

            vertex_set = item.get('vertexSet')
            sents = item.get(text_column, [])
            labels = item.get(rel_column, [])

            current_labels=[]

            for label in labels:

                head_id = label['h']
                tail_id = label['t']
                relation = rel_info[label['r']]

                current_index = 0
                head_data = None
                tail_data = None

                for sublist in vertex_set:
                      if current_index == head_id:
                          head_data = sublist
                      current_index += 1

                current_index = 0

                for sublist in vertex_set:
                      if current_index == tail_id:
                          tail_data = sublist
                      current_index += 1


                head_name = head_data[0]['name'] if head_data else None
                tail_name = tail_data[0]['name'] if tail_data else None

                true_labels.append(f'{head_name} <> {relation} <> {tail_name}')
                current_labels.append(f'{head_name} <> {relation}')
            grouped_labels.append(current_labels)
            result = " ".join(string for sublist in  sents for string in sublist)
            texts_by_line.append(result)

        return texts_by_line, grouped_labels, true_labels

    def process_predictions(self, predictions):

        preds = []

        for predict in predictions:
          for pred_ in predict:
            label = pred_.get('label')
            tail = pred_.get('text')
            result = label + ' <> ' + tail
            preds.append(result)

        return preds

    def compute_f_score(self, predicts, true_labels):

      true_set = set(true_labels)
      pred_set = set(predicts)

      tp = len(true_set.intersection(pred_set))
      fp = len(pred_set - true_set)
      fn = len(true_set - pred_set)

      precision = tp / (tp + fp) if tp + fp > 0 else 0
      recall = tp / (tp + fn) if tp + fn > 0 else 0
      f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

      return precision, recall, f1, tp, fp, fn

    def __call__(self, texts, labels, threshold=0.5):
        
        predictions = []
        list_for_label = []
        for text, label_list in tqdm(zip(texts, labels), total=len(labels), desc='Predicting'):
            for label in label_list:
                list_for_label = []
                list_for_label.append(label)
                prediction = self.model.predict_entities(text, list_for_label, threshold)
                predictions.append(prediction)

        return predictions

    def evaluate(self, dataset, rel_info, labels=None, *args, **kwargs):

          test_texts, labels, true_labels = self.prepare_dataset(dataset, rel_info)
          predictions = self.__call__(test_texts, labels)
          preds = self.process_predictions(predictions)
          return self.compute_f_score(preds, true_labels)

with open('train_annotated.json', 'r', buffering=1) as file:
    data = json.load(file)

with open('rel_info.json', 'r') as file:
    rel_info = json.load(file)

max_examples = 1000
data = data[:max_examples]

rel = DocredReGLiNEREvaluator(model_id = 'knowledgator/gliner-multitask-large-v0.5')
rel.evaluate(data, rel_info)