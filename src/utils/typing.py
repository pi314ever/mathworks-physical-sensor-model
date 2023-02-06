from typing import TypedDict, Tuple

class HashDictType(TypedDict):
    K: tuple[float,...]
    P: tuple[float,...]
    split: str

paramType = Tuple[float,...]