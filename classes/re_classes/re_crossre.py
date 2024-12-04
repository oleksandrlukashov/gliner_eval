from datasets import load_dataset
from tqdm import tqdm
from gliner import GLiNER
from classes.parent_class import AbstractEvaluator
import re

class CrossReGLiNEREvaluator(AbstractEvaluator):

    def prepare_dataset(self, dataset, text_column='sentence', rel_column='relations', *args, **kwargs):

        dataset = dataset.filter(lambda example: example[rel_column] != [])
        test_dataset = dataset['test']

        senteces = test_dataset[text_column]
        rels = test_dataset[rel_column]

        keys_to_extract = ['id_1-start', 'id_1-end', 'id_2-start', 'id_2-end', 'relation-type']
        grouped_labels = []
        true_labels = []
        texts_by_line = []

        for texts_lists, relations_lists in zip(senteces, rels):

            current_labels = []

            for relations_dicts in relations_lists:
                  
                  id_1_start = relations_dicts.get('id_1-start', None)
                  id_1_end = relations_dicts.get('id_1-end', None)
                  id_2_start = relations_dicts.get('id_2-start', None)
                  id_2_end = relations_dicts.get('id_2-end', None)
                  relation_type = relations_dicts.get('relation-type', None)
                  label = ' '.join(texts_lists[id_1_start:id_1_end+1]) + ' <> ' + relation_type
                  current_labels.append(label)
                  true_label = ' '.join(texts_lists[id_1_start:id_1_end+1]) + ' <> ' + relation_type + ' <> ' + ' '.join(texts_lists[id_2_start:id_2_end+1])
                  true_labels.append(true_label)

            text = ' '.join(texts_lists)
            text = re.sub(r'\s([?.!,;])', r'\1', text)

            texts_by_line.append(text)
            grouped_labels.append(current_labels)
            texts_by_line = [x for x in texts_by_line if x != []]
            grouped_labels = [x for x in grouped_labels if x != []]

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
        for text, label_list in tqdm(zip(texts, labels), total=len(texts), desc='Predicting'):
            for label in label_list:
                list_for_label = []
                list_for_label.append(label)
                prediction = self.model.predict_entities(text, list_for_label, threshold)
                predictions.append(prediction)

        return predictions

    def evaluate(self, dataset_id, topic, labels=None, *args, **kwargs):

          dataset = load_dataset(dataset_id, topic)
          test_texts, labels, true_labels = self.prepare_dataset(dataset)
          predictions = self.__call__(test_texts, labels)
          preds = self.process_predictions(predictions)
          return self.compute_f_score(preds, true_labels)


rel = CrossReGLiNEREvaluator(model_id = 'knowledgator/gliner-multitask-large-v0.5')
rel.evaluate('DFKI-SLT/cross_re', 'ai')

