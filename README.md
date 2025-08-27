# randomstatsmodels

A tiny, modern Python package skeleton for experimenting with forecasting and statistics utilities.

## Quick start
```bash
# from the project root
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"  # install in editable mode with dev extras
randomstatsmodels --version
```

## Usage
```python
from randomstatsmodels.metrics import mae, rmse

y_true = [1, 2, 3]
y_pred = [1.1, 1.9, 3.2]
print(mae(y_true, y_pred))
```

## Testing
```bash
pytest -q
```

## Using your models
Put your custom models in `randomstatsmodels/user_models.py` (we copied your uploaded file there).

### Python API
```python
from randomstatsmodels.api import predict
# Model ref can be 'randomstatsmodels.user_models:MyModel' or an object/callable
yhat = predict("randomstatsmodels.user_models:MyModel", X_dataframe)
```

### CLI
```bash
randomstatsmodels predict \        --model randomstatsmodels.user_models:MyModel \        --input data.csv \        --output preds.csv
```

The adapter accepts:
- Objects with `.predict(X, **kwargs)`
- Callables like `def f(X): ...`
- Classes that can be instantiated without args and have `.predict(X)`
