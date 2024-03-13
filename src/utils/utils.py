import importlib.util
import os
import signal
import sys

from omegaconf import OmegaConf


# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError


# The decorator to apply timeout to functions with a default return value on timeout
def timeout(seconds, default_value=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                return default_value
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


# Note: run from the root directory of the project
def load_config(file_path="configs/default.py"):
    """
    Load config file and transform it into the format of wandb. For wandb recording.
    """

    # Add the file's directory to the Python path
    dir_path = os.path.dirname(file_path)
    sys.path.append(dir_path)

    # Load the module
    module_name = os.path.basename(file_path).replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Extract the variables
    config = {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if not name.startswith("__")
    }

    # eliminate all modules in config if any
    for key in list(config.keys()):
        if type(config[key]) == type(config_module):
            del config[key]

    # return config
    return OmegaConf.create(config)


if __name__ == "__main__":
    config = load_config()
    # print(config.pretty())
