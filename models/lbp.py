from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from datasets.lbp import LocalBinaryPatternsDataset, LocalBinaryPatternsImageClassificationDataset
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
    def __init__(self, estimators: Optional[List] = None, scaler: Optional = None):
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

    def __str__(self) -> str:
        return (
            f"LocalBinaryPatternsClassifierBackend("
            f"train_set={self.train_set}, "
            f"distances={list(self.distances.keys())}, "
            f"estimators={self.estimators}, "
            f"scaler={self.scaler})"
        )

    def train(
            self,
            train_data: Union[LocalBinaryPatternsDataset, LocalBinaryPatternsImageClassificationDataset],
            **kwargs
    ):
        print(">> Training Local Binary Patterns Distance-Based Classifier and Estimators\n" + "=" * 50)
        print(f"> Distance Functions: {self.distances.keys()}")
        print(f"> {'Estimators:'.ljust(len('Distance Functions:'))} {self.estimators}")
        print(f"> {'Scaler:'.ljust(len('Distance Functions:'))} {self.scaler}")
        print(f"> {'Samples:'.ljust(len('Distance Functions:'))} {len(train_data)}")
        print("-" * 50)
        self.train_set = train_data
        self.train_set.lbp_vectors = self.scaler.fit_transform(self.train_set.lbp_vectors)
        for estimator in self.estimators:
            print(f"> Training {estimator}")
            estimator.fit(self.train_set.lbp_vectors, self.train_set.labels)
        print("=" * 50)

    def predict(self, data: np.ndarray, k: int = 1, distance_func: str = 'euclidean') -> int:
        data = self.scaler.transform([data])[0]
        distances = [self.distances[distance_func](data, vector) for vector in self.train_set.lbp_vectors]
        nearest_labels = [self.train_set.labels[i] for i in np.argsort(distances)[:k]]
        return np.argmax(np.bincount(nearest_labels))

    def test(
            self,
            test_data: Union[LocalBinaryPatternsDataset, LocalBinaryPatternsImageClassificationDataset],
            k: int = 1,
            save_metrics: bool = True,
            save_dir: str = "metrics"
    ) -> pd.DataFrame:
        print(f">> Evaluating on {len(test_data)} samples")

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
            print("=" * 50)
            print("> Method: Distance-Based")
            print(f"Distance Function: {distance_func}")
            print(f"Confusion Matrix: \n{cm}")
            print(f"{'Accuracy'.ljust(len('Precision'))} = {acc}")
            print(f"Precision = {prec}")
            print(f"{f'Recall'.ljust(len('Precision'))} = {rec}")
            print(f"{f'F1 Score'.ljust(len('Precision'))} = {f1}")
            metric.loc[len(metric)] = ['Distance-Based', distance_func, cm, acc, prec, rec, f1]
        for estimator in self.estimators:
            y_true = test_data.labels
            y_pred = estimator.predict(self.scaler.transform(test_data.lbp_vectors))
            cm = confusion_matrix(y_true, y_pred)
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average='macro')
            rec = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            print("-" * 50)
            print(f"> Method: {estimator}")
            print(f"Confusion Matrix: \n{cm}")
            print(f"{'Accuracy'.ljust(len('Precision'))} = {acc}")
            print(f"Precision = {prec}")
            print(f"{f'Recall'.ljust(len('Precision'))} = {rec}")
            print(f"{f'F1 Score'.ljust(len('Precision'))} = {f1}")
            metric.loc[len(metric)] = [str(estimator), None, cm, acc, prec, rec, f1]

        if save_metrics:
            metric.to_csv(increment_path(save_dir) / "metrics.csv", index=False)
            print("-" * 50)
            print(f"==>> Metrics saved to {save_dir}")
            print("=" * 50)

        return metric


if __name__ == '__main__':
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier

    backend = LocalBinaryPatternsClassifierBackend(
        estimators=[RandomForestClassifier(), SVC()],
        scaler=StandardScaler()
    )
    print(backend)
