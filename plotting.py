from matplotlib import pyplot as plt
import json
import numpy as np

num_robots = 10
num_health_stations = 3
num_hs_bays = 3

# import data


# get treasure data from each robot in data
def get_treasures(data, n_rob, n_hs, n_b):
    data = json.load(open(f'{n_rob}_{n_hs}_{n_b}.json', 'r'))


# plot treasure by robot over time
def plot_treasure(timesteps, ):
    pass


# plot 1 hs 1 bay for all robots
for r in range(0, num_robots):
    data = json.load(open(f'{r+1}_{2}_{3}.json', 'r'))

    timesteps = np.array([robot['timestep'] for robot in data[f'Robot 0']])
    treasures = np.zeros(timesteps.shape)

    # get treasures per robot over time
    for j in range(0, r + 1):
        treasures += np.array([robot['treasures'] for robot in data[f'Robot {j}']])
        
    treasures = treasures / (r+1)

    plt.plot(timesteps, treasures, label=f'{r+1} Robots')
        

plt.legend()
plt.show()
