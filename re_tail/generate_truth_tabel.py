import json

def generate_fewrel_ground_truth(input_file, output_file):
    ground_truth = {}

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)

            head = item["head"]["text"].lower()
            tail = item["tail"]["text"].lower()

            correct_relations = [relation.lower() for relation in item["names"]]
            relation_key = f"{head} <> {tail}"

            if relation_key not in ground_truth:
                ground_truth[relation_key] = []

            for relation in correct_relations:
                ground_truth[relation_key].append({
                    "head_relation": f"{head} <> {relation}",
                    "tail": tail,
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=4, ensure_ascii=False)

    print(f"ground truth saved to {output_file}")

generate_fewrel_ground_truth('val_wiki-2.json', 'ground_truth_.json')