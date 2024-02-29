# EPH: Ensembling Prioritized Hybrid Policies for Multi-agent Pathfinding

## Configurations
We made the configuration loading dynamic, so multiple configurations are allowed for different experiments under [configs/](configs/).

Before running _any_ script, you can change which configuration to load by changing the `CONFIG_NAME` variable in the [config.py](config.py) file:
```python
CONFIG_NAME = 'eph'
```
For example, the above will load the default configuration file [configs/eph.py](configs/eph.py).



### Changing model
To change the model, we made sure that the model path is loaded from the configuration file.

You can change the target by:
```
model_target = "model.Network"
```

This will load the `Network` class from the `model.py` module.


## Data generation
Go to [src/data/](src/data/) and follow the instructions in the [README.md](src/data/README.md) for generating the MovingAI's test set.


## Requirements
Install the required packages using the following command:
```bash
pip install -r requirements.txt
```