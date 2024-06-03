from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from dataset import load_dataset


@dataclass
class Row:
    label: int
    lbp_vector: np.ndarray


def calculate_euclid_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def calculate_manhattan_distance(v1, v2):
    return np.sum(np.abs(v2 - v1))


def calculate_cosine_similarity(v1, v2):
    return 1 - (np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def predict_by_lbp_vec(v: np.ndarray, rows: List[Row], dis_func=calculate_euclid_distance) -> int:
    distances = []
    for row in rows:
        distances.append(dis_func(v, row.lbp_vector))
    return rows[np.argmin(distances)].label


def evaluation(train_data: List[Row], test_data: List[Row], dis_func=calculate_euclid_distance):
    y_true = [test_row.label for test_row in test_data]
    y_pred = []
    for test_row in test_data:
        y_pred.append(predict_by_lbp_vec(test_row.lbp_vector, train_data, dis_func=dis_func))
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print("\n== Value: ")
    print(f"True Value: \t{y_true}")
    print(f"Pred Value: \t{y_pred}")
    print("\n== Metrics: ")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: \t{acc}")
    print(f"Precision: \t{prec}")
    print(f"Recall: \t{rec}")
    print(f"F1 Score: \t{f1}")


if __name__ == "__main__":
    train_set = load_dataset("./output/train.csv")
    test_set = load_dataset("./output/test.csv")
    evaluation(train_data=train_set, test_data=test_set, dis_func=calculate_manhattan_distance)
