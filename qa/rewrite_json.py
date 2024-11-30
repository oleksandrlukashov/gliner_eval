import json

with open('pred.json', 'r') as f:
    predictions = json.load(f)

predictions_dict = {entry["id"]: entry["prediction_text"] for entry in predictions}

print(predictions_dict)

file_path = 'predictions_output.json'

with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(predictions_dict, json_file, ensure_ascii=False, indent=4)
