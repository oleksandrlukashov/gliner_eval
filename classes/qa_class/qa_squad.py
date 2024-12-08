import json
from gliner import GLiNER
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parent_class import AbstractEvaluator
from tqdm import tqdm

class QaGLiNEREvaluator(AbstractEvaluator):

    def evaluate(self, dataset, labels=None, *args, **kwargs):

        context, questions, ans, impossibles = self.prepare_dataset(dataset)
        predictions = self.__call__(context, questions)
        preds = self.process_predictions(predictions)
        return self.compute_f_score(preds, ans, impossibles)

    def prepare_dataset(self, dataset, *args, **kwargs):

        contexts = []
        current_context = []
        current_questions = []
        questions = []
        answers__ = []
        answers = []
        impossibles = []
        for dicts in dataset:
          paragraphs = dicts.get('paragraphs')
          for p in paragraphs:
            context = p.get('context')
            contexts.append(context)
            qas = p.get('qas')
            question = [d['question'] for d in qas if 'question' in d]
            answers_ = [k['answers'] for k in qas if 'answers' in k]
            ans = [list({d['text'] for d in sublist}) for sublist in answers_]
            is_impossible = [m['is_impossible'] for m in qas if 'is_impossible' in m]
            impossibles.append(is_impossible)
            answers.append(ans)
            questions.append(question)

        return contexts, questions, answers, impossibles

    def __call__(self, contexts, questions, threshold=0.5):

        grouped_preds = []
        labels = ['answer']
        for context, question in tqdm(zip(contexts, questions), total=len(contexts), desc="Processing contexts"):
            predictions = []
            for q in question:
              input_text = q + "\n" + context
              prediction = self.model.predict_entities(input_text, labels, threshold=0.5)
              predictions.append(prediction)

            grouped_preds.append(predictions)

        return grouped_preds

    def compute_f_score(self, predicts, true_labels, pos):
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for predict, ans, posibl in zip(predicts, true_labels, pos):
            for pred, ans_, pos_ in zip(predict, ans, posibl):
                if pos_ == False:
                    if pred:
                        matched = any(p == a for p in pred for a in ans_)
                        if matched:
                            true_positive += 1
                        else:
                            false_positive += 1
                    else:  # Если предсказание пустое, а ответ возможен
                        false_negative += 1
                elif pos_ == True:
                    if pred == []:
                        true_positive += 1
                    else:
                        false_positive += 1


        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }



    def process_predictions(self, predictions):

        result = [[[item['text'] for item in inner_list if 'text' in item]
        for inner_list in outer_list] for outer_list in predictions]

        return result

with open('data.json') as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']