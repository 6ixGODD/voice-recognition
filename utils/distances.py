import numpy as np


def euclidean_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.linalg.norm(vector1 - vector2)


def manhattan_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.sum(np.abs(vector2 - vector1))


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return 1 - (np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def minkowski_distance(vector1, vector2, p=3) -> float:
    return np.sum(np.abs(vector1 - vector2) ** p) ** (1 / p)


def chebyshev_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.max(np.abs(vector1 - vector2))


def bray_curtis_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    return np.sum(np.abs(vector1 - vector2)) / np.sum(np.abs(vector1 + vector2))
