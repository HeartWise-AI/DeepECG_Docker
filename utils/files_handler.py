import os
import csv
import json

def save_to_csv(metrics: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    # Open the file and create a CSV writer
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for metric in metrics:
            for value in metrics[metric]:
                writer.writerow([metric, value, metrics[metric][value]])

def save_to_json(data: dict, path: str) -> None:
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def read_api_key(path: str) -> dict[str, str]:
    with open(path) as f:
        api_key = json.load(f)
    return api_key
