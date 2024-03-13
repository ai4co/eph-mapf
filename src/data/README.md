# Map data generation

## Random maps

We use a custom data generator to generate the random maps in which we make sure that goals are reachable from the start position.

To run the data generation, you may use:

```bash
python3 generate_data_random_maps.py --num_instances 100
```

```markdown
usage: generate_data_random_maps.py [-h] [--width WIDTH] [--height HEIGHT] [--density DENSITY] [--num_agents NUM_AGENTS] [--num_instances NUM_INSTANCES]

Generate random maps and agents for testing

options:
  -h, --help            show this help message and exit
  --width WIDTH         Width of the map
  --height HEIGHT       Height of the map
  --density DENSITY     Density of the map
  --num_agents NUM_AGENTS
                        Number of agents
  --num_instances NUM_INSTANCES
                        Number of instances
```

## Structured maps (MovingAI's MAPF Benchmark)
We use the data generator from [SACHA](https://github.com/Qiushi-Lin/SACHA/tree/main/benchmarks) to generate the data for the map.


The maps are taken from [Moving AI's MAPF Benchmark](https://movingai.com/benchmarks/mapf/index.html).

To run the data generation, you may use:

```bash
python3 generate_data.py --num_instances 100
```


> ![TIP]
> You may also load more maps inside of the `maps/` folder!

You can modify the [config.yaml](config.yaml) file to change the number of instances and the map size.
