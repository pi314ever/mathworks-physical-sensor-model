from typing import TypedDict, Tuple, Literal

class HashDictType(TypedDict):
    coefficients: tuple[float,...]
    type: Literal['radial', 'tangential']
    split: Literal['test', 'train', 'val']

paramType = Tuple[float,...]