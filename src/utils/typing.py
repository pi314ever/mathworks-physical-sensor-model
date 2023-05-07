from typing import TypedDict, Tuple, Literal

paramType = Tuple[float, ...]

distortionType = Literal['radial', 'tangential', 'combined']

splitType = Literal['test', 'train', 'valid']

class HashDictType(TypedDict):
    params: paramType
    distortion: distortionType
    split: splitType
