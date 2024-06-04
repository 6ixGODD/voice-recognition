from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from dataset import load_dataset
from lbp import (
    calculate_bray_curtis_distance,
    calculate_chebyshev_distance,
    calculate_cosine_similarity,
    calculate_euclid_distance,
    calculate_manhattan_distance,
    calculate_minkowski_distance,
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
    # Standard
    scalar = StandardScaler()
    vectors = np.array([row.lbp_vector for row in rows])
    vectors = scalar.fit_transform(vectors)
    v = scalar.transform([v])[0]

    distances = [dist_func(v, vector) for vector in vectors]

    nearest_labels = [rows[i].label for i in np.argsort(distances)[:N]]
    return np.argmax(np.bincount(nearest_labels))


def evaluation_(
        train_data: List[Row],
        test_data: List[Row],
        dist_func=calculate_euclid_distance,
        k: int = 1
):
    y_true = [test_row.label for test_row in test_data]
    y_pred = []
    for test_row in test_data:
        y_pred.append(predict_knn(test_row.lbp_vector, train_data, dist_func=dist_func, N=k))
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # print("\n== Value: ")
    # print(f"True Value: \t{np.array(y_true)}")
    # print(f"Pred Value: \t{np.array(y_pred)}")
    print("-- Metrics: ")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: \t{acc}")
    print(f"Precision: \t{prec}")
    print(f"Recall: \t{rec}")
    print(f"F1 Score: \t{f1}")


def evaluation_svm(train_data: List[Row], test_data: List[Row], _svm):
    X_train = np.array([row.lbp_vector for row in train_data])
    y_train = np.array([row.label for row in train_data])
    X_test = np.array([row.lbp_vector for row in test_data])
    y_test = np.array([row.label for row in test_data])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    _svm.fit(X_train, y_train)
    y_pred = _svm.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    # print("\n== Value: ")
    # print(f"True Value: \t{y_test}")
    # print(f"Pred Value: \t{y_pred}")

    print("-- Metrics: ")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: \t{acc}")
    print(f"Precision: \t{prec}")
    print(f"Recall: \t{rec}")
    print(f"F1 Score: \t{f1}")


def evaluation_rf(train_data: List[Row], test_data: List[Row], _rf):
    X_train = np.array([row.lbp_vector for row in train_data])
    y_train = np.array([row.label for row in train_data])
    X_test = np.array([row.lbp_vector for row in test_data])
    y_test = np.array([row.label for row in test_data])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    _rf.fit(X_train, y_train)
    y_pred = _rf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

    # print("\n== Value: ")
    # print(f"True Value: \t{y_test}")
    # print(f"Pred Value: \t{y_pred}")

    print("-- Metrics: ")
    print(f"Confusion Matrix: \n{cm}")
    print(f"Accuracy: \t{acc}")
    print(f"Precision: \t{prec}")
    print(f"Recall: \t{rec}")
    print(f"F1 Score: \t{f1}")


if __name__ == "__main__":
    import time

    _train_data = load_dataset("./output-spectrogram-flatten/train.csv")
    _test_data = load_dataset("./output-spectrogram-flatten/test.csv")
    K = 1
    start = time.time()
    print("== KNN ==")
    print("\n-- Euclid Distance --")
    evaluation_(
        train_data=_train_data,
        test_data=_test_data,
        dist_func=calculate_euclid_distance,
        k=K
    )
    print("\n-- Manhattan Distance --")
    evaluation_(
        train_data=_train_data,
        test_data=_test_data,
        dist_func=calculate_manhattan_distance,
        k=K
    )
    print("\n-- Cosine Similarity --")
    evaluation_(
        train_data=_train_data,
        test_data=_test_data,
        dist_func=calculate_cosine_similarity,
        k=K
    )
    print("\n-- Minkowski Distance --")
    evaluation_(
        train_data=_train_data,
        test_data=_test_data,
        dist_func=calculate_minkowski_distance,
        k=K
    )
    print("\n-- Chebyshev Distance --")
    evaluation_(
        train_data=_train_data,
        test_data=_test_data,
        dist_func=calculate_chebyshev_distance,
        k=K
    )
    print("\n-- Bray Curtis Distance --")
    evaluation_(
        train_data=_train_data,
        test_data=_test_data,
        dist_func=calculate_bray_curtis_distance,
        k=K
    )

    print("\n== SVM ==")
    from sklearn.svm import SVC

    svm = SVC(kernel='linear', C=1.0, random_state=42, probability=True)
    evaluation_svm(
        train_data=_train_data,
        test_data=_test_data,
        _svm=svm
    )

    print("\n== Random Forest ==")
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    evaluation_rf(
        train_data=_train_data,
        test_data=_test_data,
        _rf=rf
    )
    print("Time: {:.2f}s".format(time.time() - start))
