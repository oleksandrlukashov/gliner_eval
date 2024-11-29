from datasets import load_dataset
import json


def calculate_f1(prediction, ground_truth):

    pred_tokens = prediction.split()
    gt_tokens = ground_truth.split()

    common_tokens = set(pred_tokens) & set(gt_tokens)
    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_qa(predictions, dataset, score_threshold=0.9):

    total_em = 0
    total_f1 = 0
    count = 0

    for i, example in enumerate(dataset):

        ground_truths = example["answers"]["text"]

        example_predictions = predictions.get(str(i), [])
        filtered_predictions = [
            pred for pred in example_predictions if pred["score"] >= score_threshold
        ]

        if not filtered_predictions:
            continue

        predicted_answer = max(filtered_predictions, key=lambda x: x["score"])["answer"]

        em = 0
        f1 = 0
        for gt in ground_truths:
            em = max(em, int(predicted_answer.strip() == gt.strip()))
            f1 = max(f1, calculate_f1(predicted_answer, gt))

        total_em += em
        total_f1 += f1
        count += 1


    em_score = total_em / count if count > 0 else 0.0
    f1_score = total_f1 / count if count > 0 else 0.0

    return em_score, f1_score

dataset = load_dataset("squad_v2", split="validation")

file_path_1 = 'eval/qa/question_answer_results_20241128_165608.json'
with open(file_path_1, 'r') as f:
    predictions = json.load(f)

em, f1 = evaluate_qa(predictions, dataset, score_threshold=0.9)
print(f"Exact Match (EM): {em:.2%}")
print(f"F1 Score: {f1:.2%}")
