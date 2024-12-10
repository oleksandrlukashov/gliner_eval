import json
from gliner import GLiNER
from tqdm import tqdm
import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parent_class import AbstractEvaluator

class QaGLiNEREvaluator(AbstractEvaluator):

    def evaluate(self, dataset, labels=None, *args, **kwargs):

        context, questions, ans, impossibles, ids = self.prepare_dataset(dataset)
        predictions = self.__call__(context, questions, ids)
        return self.compute_f_score(predictions, dataset)

    def prepare_dataset(self, dataset, *args, **kwargs):

        contexts = []
        current_context = []
        current_questions = []
        questions = []
        answers__ = []
        answers = []
        impossibles = []
        ids = []
        for dicts in dataset:
          paragraphs = dicts.get('paragraphs')
          for p in paragraphs:
            context = p.get('context')
            contexts.append(context)
            qas = p.get('qas')
            question = [d['question'] for d in qas if 'question' in d]
            answers_ = [k['answers'] for k in qas if 'answers' in k]
            qid = [id['id'] for id in qas if 'id' in id]
            ans = [list({d['text'] for d in sublist}) for sublist in answers_]
            is_impossible = [m['is_impossible'] for m in qas if 'is_impossible' in m]
            impossibles.append(is_impossible)
            answers.append(ans)
            questions.append(question)
            ids.append(qid)

        return contexts, questions, answers, impossibles, ids

    def __call__(self, contexts, questions, ids, threshold=0.5):

        labels = ['answer']
        predictions = {}
        for context, question, id in tqdm(zip(contexts, questions, ids), total=len(contexts), desc="Processing contexts"):
            for q, i in zip(question, id):
              input_text = q + "\n" + context
              prediction = self.model.predict_entities(input_text, labels, threshold=0.5)
              if prediction:
                  best_prediction = max(prediction, key=lambda x: x.get('score', 0))
                  predictions[i] = best_prediction.get('text', "")
              else:
                  predictions[i] = ""

        return predictions

    def compute_f_score(self, preds, dataset):

        def make_qid_to_has_ans(dataset):
          qid_to_has_ans = {}
          for article in dataset:
            for p in article['paragraphs']:
              for qa in p['qas']:
                qid_to_has_ans[qa['id']] = bool(qa['answers'])
          return qid_to_has_ans

        def normalize_answer(s):
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

        def get_raw_scores(dataset, preds):
          exact_scores = {}
          f1_scores = {}
          for article in dataset:
            for p in article['paragraphs']:
              for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                  gold_answers = ['']
                if qid not in preds:
                  print('Missing prediction for %s' % qid)
                  continue
                a_pred = preds[qid]
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
          return exact_scores, f1_scores

        def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh=0):
          new_scores = {}
          for qid, s in scores.items():
            pred_na = na_probs[qid] > na_prob_thresh
            if pred_na:
              new_scores[qid] = float(not qid_to_has_ans[qid])
            else:
              new_scores[qid] = s
          return new_scores

        def make_eval_dict(exact_scores, f1_scores, qid_list=None):
          if not qid_list:
            total = len(exact_scores)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores.values()) / total),
                ('f1', 100.0 * sum(f1_scores.values()) / total),
                ('total', total),
            ])
          else:
            total = len(qid_list)
            return collections.OrderedDict([
                ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ('total', total),
            ])

        def merge_eval(main_eval, new_eval, prefix):
          for k in new_eval:
            main_eval['%s_%s' % (prefix, k)] = new_eval[k]

        def get_raw_scores(dataset, preds):
          exact_scores = {}
          f1_scores = {}
          for article in dataset:
            for p in article['paragraphs']:
              for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                  gold_answers = ['']
                if qid not in preds:
                  print('Missing prediction for %s' % qid)
                  continue
                a_pred = preds[qid]
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
          return exact_scores, f1_scores

        na_probs = {k: 0.0 for k in preds}
        qid_to_has_ans = make_qid_to_has_ans(dataset)
        has_ans_qids = [k for k, v in qid_to_has_ans.items() if v]
        no_ans_qids = [k for k, v in qid_to_has_ans.items() if not v]
        exact_raw, f1_raw = get_raw_scores(dataset, preds)
        out_eval = make_eval_dict(exact_raw, f1_raw)
        has_ans_eval = make_eval_dict(exact_raw, f1_raw, qid_list=has_ans_qids)
        merge_eval(out_eval, has_ans_eval, 'HasAns')
        no_ans_eval = make_eval_dict(exact_raw, f1_raw, qid_list=no_ans_qids)
        merge_eval(out_eval, no_ans_eval, 'NoAns')
        return out_eval



    def process_predictions(self, predictions):

        result_list = [{"id": key, "prediction_text": value} for key, value in predictions.items()]
        json_result = json.dumps(result_list, indent=2, ensure_ascii=False)
        return json_result

with open('data.json') as f:
    dataset_json = json.load(f)
    dataset = dataset_json['data']

