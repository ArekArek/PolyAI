# How to run
Modify config `config.yml`

## Generate training and test dataset:
```
uv run generator.py --training
uv run generator.py --test
```
Logs are saved under `logs/<run_date>-generate.log`.
Training data is saved under `training-data/` and `test-data/`.

## Train model
```
uv run train_GRU.py
```
Logs are saved under `logs/<run_date>-train.log`.
Trained models are saved under `model-output/<run_date>/`. 
Best model is named `best_model.h5`.
Each model is an result of single epoch. This epoch with its loss is encoded in model name: `model_<epoch>_<loss>.h5`

## Evaluate model
```
uv run evaluate.py -d training-data/ -m model-output/<run_date>/best_model.h5
```

## Present model
```
uv run show.py -d training-data/ -m model-output/<run_date>/best_model.h5
```



