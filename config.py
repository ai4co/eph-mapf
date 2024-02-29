######################################
# Only need to set the config name here to load it
# CONFIG_NAME = "eph-base"
CONFIG_NAME = "eph"
######################################

from utils import load_config

config = load_config(f"configs/{CONFIG_NAME}.py")
# Testing
if __name__ == '__main__':
    import rich
    from omegaconf import OmegaConf
    
    print(dict(config))

    rich.print(OmegaConf.to_yaml(config))
