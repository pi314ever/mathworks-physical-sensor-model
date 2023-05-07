

def validate_split(fn):
    def wrapper(*args, **kwargs):
        split = kwargs.get('split', None)
        if split is None:
            raise ValueError('split must be specified')
        if split not in ['train', 'valid', 'test']:
            raise ValueError(f'Invalid split: {split}')
        return fn(*args, **kwargs)
    return wrapper

def validate_distortion(fn):
    def wrapper(*args, **kwargs):
        distortion = kwargs.get('distortion', None)
        if distortion is None:
            raise ValueError('distortion must be specified')
        if distortion not in ['radial', 'tangential', 'combined']:
            raise ValueError(f'Invalid distortion: {distortion}')
        return fn(*args, **kwargs)
    return wrapper

def validate_model_type(fn):
    def wrapper(*args, **kwargs):
        model_type = kwargs.get('model_type', None)
        if model_type is None:
            raise ValueError('model_type must be specified')
        if model_type not in ['radial', 'tangential', 'combined']:
            raise ValueError(f'Invalid model_type: {model_type}')
        return fn(*args, **kwargs)
    return wrapper