from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Row:
    label: int
    lbp_vector: np.ndarray


def load_dataset(data_csv: str) -> List[Row]:
    data = np.genfromtxt(data_csv, delimiter=",")
    row_list = []
    for row in data:
        row_list.append(Row(label=int(row[0]), lbp_vector=row[1:]))
    return row_list
