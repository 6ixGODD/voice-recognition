from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from datasets.lbp import LocalBinaryPatternsDataset, LocalBinaryPatternsImageClassifierDataset
from models.common import ClassifierBackend
from utils.distances import (
    bray_curtis_distance,
    chebyshev_distance,
    cosine_similarity,
    euclidean_distance,
    manhattan_distance,
    minkowski_distance,
)
from utils.files import increment_path


class LocalBinaryPatternsClassifierBackend(ClassifierBackend):
    def __init__(self, estimators=None, scaler=None):
        if estimators is None:
            estimators = []
        if scaler is None:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        self.train_set = None
        self.train_set: LocalBinaryPatternsDataset
        self.test_set: LocalBinaryPatternsDataset
        self.estimators = estimators
        self.scaler = scaler
        self.distances = {
            'euclidean':   euclidean_distance,
            'manhattan':   manhattan_distance,
            'cosine':      cosine_similarity,
            'minkowski':   minkowski_distance,
            'chebyshev':   chebyshev_distance,
            'bray_curtis': bray_curtis_distance,
        }

    def __str__(self):
        return (
            f"Local Binary Patterns Classifier \n"
            f"Estimators: {self.estimators} \n"
            f"Scaler: {self.scaler} \n"
            f"Distance Functions: {self.distances.keys()} \n"
            f"Train Set: {self.train_set} \n"
        )

    def train(self, train_data: Union[LocalBinaryPatternsDataset, LocalBinaryPatternsImageClassifierDataset], **kwargs):
        self.train_set = train_data
        train_data.lbp_vectors = self.scaler.fit_transform(train_data.lbp_vectors)
        for estimator in self.estimators:
            estimator.fit(train_data.lbp_vectors, train_data.labels)

    def predict(self, data: np.ndarray, k: int = 1, distance_func: str = 'euclidean') -> int:
        data = self.scaler.transform([data])[0]
        distances = [self.distances[distance_func](data, vector) for vector in self.train_set.lbp_vectors]
        nearest_labels = [self.train_set.labels[i] for i in np.argsort(distances)[:k]]
        return np.argmax(np.bincount(nearest_labels))

    def evaluate(
            self,
            test_data: LocalBinaryPatternsDataset,
            k: int = 1,
            save_metrics: bool = True,
            save_dir: str = "metrics"
    ):
        metric = pd.DataFrame(
            columns=['Method', 'Distance Function', 'Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
        )
        for distance_func in self.distances:
            y_true = test_data.labels
            y_pred = []
            for test in test_data.lbp_vectors:
                y_pred.append(self.predict(test, distance_func=distance_func, k=k))
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            print("Method: Distance-Based")
            print(f"Distance Function: {distance_func}")
            print(f"Confusion Matrix: {cm}")
            print(f"Accuracy: {acc}")
            print(f"Precision: {prec}")
            print(f"Recall: {rec}")
            print(f"F1 Score: {f1}")
            print("\n")
            metric.loc[len(metric)] = ['Distance-Based', distance_func, cm, acc, prec, rec, f1]
        for estimator in self.estimators:
            y_true = test_data.labels
            y_pred = estimator.predict(self.scaler.transform(test_data.lbp_vectors))
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            print(f"Method: {estimator.__class__.__name__}")
            print(f"Confusion Matrix: {cm}")
            print(f"Accuracy: {acc}")
            print(f"Precision: {prec}")
            print(f"Recall: {rec}")
            print(f"F1 Score: {f1}")
            print("\n")
            metric.loc[len(metric)] = [estimator.__class__.__name__, None, cm, acc, prec, rec, f1]

        if save_metrics:
            metric.to_csv(increment_path(save_dir) / "metrics.csv", index=False)


if __name__ == '__main__':
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    backend = LocalBinaryPatternsClassifierBackend(
        estimators=[RandomForestClassifier(), SVC()],
        scaler=StandardScaler()
    )
    print(backend)
