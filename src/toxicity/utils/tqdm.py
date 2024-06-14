from .notebook import in_notebook

tqdm = None

if in_notebook():
    from tqdm.notebook import tqdm as _tqdm
    tqdm = _tqdm
else:
    from tqdm import tqdm as _tqdm
    tqdm = _tqdm