import json
from gliner import GLiNER
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parent_class import AbstractEvaluator

class QaGLiNEREvaluator(AbstractEvaluator):

    def evaluate(self, dataset, labels=None, *args, **kwargs):

        context, questions, ans = self.prepare_dataset(dataset)
        predictions = self.__call__(context, questions)
        preds = self.process_predictions(predictions)
        return self.compute_f_score(preds, ans)

    def prepare_dataset(self, dataset, *args, **kwargs):

        contexts = []
        current_context = []
        current_questions = []
        questions = []
        answers__ = []
        answers = []
        for dicts in dataset:
          paragraphs = dicts.get('paragraphs')
          for p in paragraphs:
            context = p.get('context')
            contexts.append(context)
            qas = p.get('qas')
            question = [d['question'] for d in qas if 'question' in d]
            answers_ = [k['answers'] for k in qas if 'answers' in k]
            ans = [list({d['text'] for d in sublist}) for sublist in answers_]
            answers.append(ans)
            questions.append(question)

        return contexts, questions, answers
        
    def __call__(self, contexts, questions, threshold=0.5):

        grouped_preds = []
        labels = ['answer']
        for context, question in zip(contexts, questions):
            predictions = []
            for q in question:
              input_text = q + context
              prediction = self.model.predict_entities(input_text, labels, threshold=0.5)
              predictions.append(prediction)

            grouped_preds.append(predictions)

        return grouped_preds

    def compute_f_score(self, predicts, true_labels):

      true_positive = 0
      false_positive = 0
      false_negative = 0

      for predict, ans in zip(predicts, true_labels):
          for pred, ans_ in zip(predict, ans):
              pred_set, ans_set = set(pred), set(ans_)
              tp = len(pred_set & ans_set)
              true_positive += tp
              fp = len(pred_set - ans_set)
              false_positive += fp
              fn = len(ans_set - pred_set)
              false_negative += fn

      precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
      recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
      f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

      return {
          "true_positive": true_positive,
          "false_positive": false_positive,
          "false_negative": false_negative,
          "precision": precision,
          "recall": recall,
          "f1_score": f1_score,
      }

            

    def process_predictions(self, predictions):

        result = [[[item['text'] for item in inner_list if 'text' in item]
        for inner_list in outer_list] for outer_list in predictions]

        return result

with open('classes/qa_class/data.json') as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']


evaluator = QaGLiNEREvaluator('knowledgator/gliner-multitask-v1.0')
results = evaluator.evaluate(dataset)
output_file = 'evaluation_results_qa_1.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

evaluator = QaGLiNEREvaluator('knowledgator/gliner-multitask-large-v0.5')
results = evaluator.evaluate(dataset)
output_file = 'evaluation_results_qa_05.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)
