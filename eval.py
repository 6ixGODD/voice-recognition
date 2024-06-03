from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import load_dataset
from lbp import (
    calculate_cosine_similarity, calculate_euclid_distance, calculate_manhattan_distance, calculate_jaccard_similarity,
    calculate_dice_similarity, calculate_hamming_distance
)


@dataclass
class Row:
    label: int
    lbp_vector: np.ndarray


def predict_knn(
        v: np.ndarray,
        rows: List[Row],
        dist_func=calculate_euclid_distance,
        N: int = 1
) -> int:
    distances = np.array([dist_func(v, row.lbp_vector) for row in rows])
    nearest_labels = [rows[i].label for i in np.argsort(distances)[:N]]
    return np.argmax(np.bincount(nearest_labels))


def evaluation_knn(
        train_data: List[Row],
        test_data: List[Row],
        dist_func=calculate_euclid_distance,
        N: int = 1
):
    y_true = [test_row.label for test_row in test_data]
    y_pred = []
    for test_row in test_data:
        y_pred.append(predict_knn(test_row.lbp_vector, train_data, dist_func=dist_func, N=N))
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("\n== Value: ")
    print(f"True Value: \t{y_true}")
    print(f"Pred Value: \t{y_pred}")
    print("\n== Metrics: ")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: \t{acc}")
    print(f"Precision: \t{prec}")
    print(f"Recall: \t{rec}")
    print(f"F1 Score: \t{f1}")


def evaluation_svm(train_data: List[Row], test_data: List[Row], svm: SVC):
    X_train = np.array([row.lbp_vector for row in train_data])
    y_train = np.array([row.label for row in train_data])
    X_test = np.array([row.lbp_vector for row in test_data])
    y_test = np.array([row.label for row in test_data])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    print("\n== Metrics: ")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: \t{acc}")
    print(f"Precision: \t{prec}")
    print(f"Recall: \t{rec}")
    print(f"F1 Score: \t{f1}")


if __name__ == "__main__":
    import time

    start = time.time()
    print("== KNN ==")
    print("\n-- Euclid Distance --")
    evaluation_knn(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        dist_func=calculate_euclid_distance,
        # N=3
    )
    print("\n-- Manhattan Distance --")
    evaluation_knn(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        dist_func=calculate_manhattan_distance,
        # N=3
    )
    print("\n-- Cosine Similarity --")
    evaluation_knn(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        dist_func=calculate_cosine_similarity,
        # N=3
    )
    print("\n-- Jaccard Similarity --")
    evaluation_knn(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        dist_func=calculate_jaccard_similarity,
        # N=3
    )
    print("\n-- Dice Similarity --")
    evaluation_knn(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        dist_func=calculate_dice_similarity,
        # N=3
    )
    print("\n-- Hamming Distance --")
    evaluation_knn(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        dist_func=calculate_hamming_distance,
        # N=3
    )

    print("\n== SVM ==")
    svm = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    evaluation_svm(
        train_data=load_dataset("./output-flatten/train.csv"),
        test_data=load_dataset("./output-flatten/test.csv"),
        svm=svm
    )
    print(f"\nExecution Time: {time.time() - start}")
