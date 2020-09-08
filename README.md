# Scikit Learn models

Enables using external predictive models from  [Scikit Learn](http://scikit-learn.org/stable/index.html) library (for example [Random Forest Regressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)) implementation . Fitting, saving, loading and live predict is enabled. Live predictions works via Kafka stream.

Predictive model is designed in an decentralized fashion, meaning that several instances (submodels) will be created and used for each specifc sensor and horizon (# submodels = # sensors * # horiozons). Decentralized architecture enables paralelization.

#### Usage
```main.py [-h] [-c CONFIG] [-f] [-s] [-l] [-p]```

#### Optional parameters:
| Short   |      Long     |  Description |
|----------|-------------|------|
| `-h` | `--help` | show help |
| `-c CONFIG` | `--config CONFIG` | path to config file (example: `config.json`) |
| `-f` | `--fit` | Learning the model from dataset (in `/data/fused`)|
| `-s` | `--save` | save model to file |
| `-l` | `--load` | load model from file |
| `-p` | `--predict` | Start live predictions (via Kafka) |

#### Config file:
Example of config file, where the useres spcifies kafka server address, which scikit algorithm to use, prediction horizons and sesnsors for which the model will be learned/loaded/saved/predicted.

Parameters:
- **bootstrap_servers**: string (or list of �host[:port]� strings) that the consumer should contact to bootstrap initial cluster metadata
- **algorithm**: string as scikit-learn model constructor with initialization parameters
- **evaluation_periode**: define time periode (in hours) for which the model will be evaluated during live predictions (evaluations metrics added to ouput record)
- **evaluation_split_point**: define training and testing spliting point in the dataset, for model evaluation during learning phase (fit takes twice as long time)
- **prediction_horizons**: list of prediction horizons (in hours) for which the model will be trained to predict for.
- **sesnors**: list of sensors for which this specific instance will train the models and will be making predictions.

Example of config file:
```json
{
    "bootstrap_servers": "127.0.0.1:9092",
    "algorithm": "sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=16)",
    "evaluation_period": 72,
    "evaluation_split_point": 0.8,
    "prediction_horizons": [1, 2, 3],
    "sensors": ["test", "test2"]
}
```

## Running multiple instances:
Modeling instance is loosely coupled, therefore it can be started as multiple processes (simple parallelization). For this purpose, we can use `start_cluster.sh` script with the same input parameters as `main.py`. Cluster is defined in a separate config file `cluster.json`. The script runs several instances of `main.py` in a tmux session (named `modeling_cluster`), each under a different window.

#### Usage
```bash start_cluster.sh [-f] [-s] [-l] [-p]```

#### Config file:
Specify which sensors should be processed by a specific instance in a separate line.

Example of cluster config file:
```json
["N1", "N2"]
["N3", "N4"]
["N5", "N6"]
["N7", "N8"]
```

## Assumptions:
- **Training data**: all the training files should be stored in a subfolder called `/data/fused`. Data should be stored as json objects per line (e.g. `{�timestamp�: 1459926000, �ftr_vector�: [1, 2, 3]}`). Separate file for each sensor and prediction horizon. Files should be named the same as input kafka topics, that is `{sensor}_{horizon}h` (e.g. �sensor1_3h.json�)
- **Models**: all the models are stored in a subfolder called `/models`.  Each sensor and horizon has its own model. The name of the models is composed of sensor name and prediction horizon, `model_{sensor}_{horizon}h` (e.g. �model_sensor1_3h�)
- **Input kafka topic**: The names of input kafka topics on which the prototype is listening for live data should be in the same format as trainng data file names, that is `features_{sensor}_{horizon}h`.
- **Output kafka topic**: Predictions are sent on different topics based on a sensor names, that is `{sensor}` (e.g. �sensor1�).

## Examples:

#### Fiting models
```python main.py -f```

#### Firing and saving models
```python main.py -f -s```

#### Loading models
```python main.py -l```

#### Start live predictions (via Kafka)
```python main.py -l -p```

#### Start modeling cluster
```bash start_cluster.sh  -l -p```

## Requirements

- python 2.7.x
- kafka-python-1.3.4
- Anaconda2-4.4.0
