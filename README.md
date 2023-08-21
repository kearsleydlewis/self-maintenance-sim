# An Autonomic Architecture for Multi-Agent Self-Maintaining Robotic Systems

This repo contains the supporting simulation code for "An Autonomic Architecture for Multi-Agent Self-Maintaining Robotic Systems".

### How to run

Version Info: python 3.8.5

1. Clone the repository

2. `python simulation.py --data_output`

```
usage: Self-Maintenance Simulation [-h] [-s MAP_SIZE] [-t MAX_TIME] [-dt TIME_STEP] [--real_time] [--data_output]

optional arguments:
  -h, --help            show this help message and exit
  -s MAP_SIZE, --map_size MAP_SIZE
                        Size of the square map in meters (default = 20m).
  -t MAX_TIME, --max_time MAX_TIME
                        Maximum time of the simulation (default = 600s).
  -dt TIME_STEP, --time_step TIME_STEP
                        Size of simulation time step (default = 0.2s).
  --real_time           Run the simulation in real time, which will show plot in real time (unless there are too many
                        robots/hs, then it will lag a bit).
  --data_output         Generate the data for plotting instead of running simulation
```

This command will generate 90 (10 * 3 * 3) JSON files in the format `{number of robots}_{number of health stations}_{number of bays at each health station}.json`. These files contain the state at each step in the simulation, which can be used by the plotting script to generate the plots for the paper.

3. Using generated JSON values, generate the plots with `python plotting.py`


##### Version Info

- python 3.8.5
- NumPy 1.19.2
- Matplotlib 3.3.2