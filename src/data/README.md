# Map data generation

We use the data generator from [SACHA](https://github.com/Qiushi-Lin/SACHA/tree/main/benchmarks) to generate the data for the map.


The maps are taken from [Moving AI's MAPF Benchmark](https://movingai.com/benchmarks/mapf/index.html).

To run the data generation, you may use:

```bash
python3 generate_data.py --num_instances 300
```


> ![TIP]
> You may also load more maps inside of the `maps/` folder!