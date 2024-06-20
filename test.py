import json


def save_metrics(metric):
    try:
        with open('./predictions/metrics.json', 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(metric)

    with open('./predictions/metrics.json', 'w') as f:
        json.dump(data, f, indent=2)


metric = {
    "Stock": "AUDI",
    "Train": {
        "MAE": 47.41729390974235,
        "MSE": 3743.4709100411,
        "RMSE": 61.1839105487799,
        "R2": 0.9416392474043404
    },
    "Test": {
        "MAE": 95.91094249560479,
        "MSE": 13324.300936867598,
        "RMSE": 115.43093578788833,
        "R2": -1.6029832480677753
    }
}

save_metrics(metric)
