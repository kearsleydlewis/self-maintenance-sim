from matplotlib import pyplot as plt
import json
import numpy as np
import argparse

# max number of generated data
num_robots = 10
num_health_stations = 3
num_hs_bays = 3


variable = 'treasures'
# plot treasure over time
# plot 1 hs 1 bay for all robots
for r in range(0, num_robots):
    data = json.load(open(f'{r+1}_{1}_{1}.json', 'r'))

    timesteps = np.array([robot['timestep'] for robot in data[f'Robot 0']])
    treasures = np.zeros(timesteps.shape)

    # get treasures per robot over time
    for j in range(0, r + 1):
        treasures += np.array([robot[variable] for robot in data[f'Robot {j}']])
        
    treasures = treasures / (r+1)

    plt.plot(timesteps, treasures, label=f'{r+1} Robots')

plt.title('Average # of Treasures Collected over Time (s)\nper # of Robots with 1 Health Station with 1 Bay')
plt.xlabel('Time (s)')
plt.ylabel('# of Treasures')
plt.legend()
plt.show()

# plot all hs/bays per robot
r = 9
for hs in range(0, num_health_stations):
    for bay in range(0, num_hs_bays):
        data = json.load(open(f'{r+1}_{hs+1}_{bay+1}.json', 'r'))

        timesteps = np.array([robot['timestep'] for robot in data[f'Robot 0']])
        treasures = np.zeros(timesteps.shape)

        # get treasures per robot over time
        for j in range(0, r + 1):
            treasures += np.array([robot[variable] for robot in data[f'Robot {j}']])
            
        treasures = treasures / (r+1)

        plt.plot(timesteps, treasures, label=f'{hs+1} HS, {bay+1} Bays')

plt.title(f'Average # of Treasures Collected over Time (s)\nper # of Health Stations and # of Bays with {r+1} Robots')
plt.xlabel('Time (s)')
plt.ylabel('# of Treasures')
plt.legend()
plt.show()

# plot task time as percentage of total time
variable = 'task_up_time'
for r in range(0, num_robots):
    data = json.load(open(f'{r+1}_{1}_{1}.json', 'r'))

    timesteps = np.array([robot['timestep'] for robot in data[f'Robot 0']])[1:]
    states = np.zeros(timesteps.shape)
    

    # get treasures per robot over time
    for j in range(0, r + 1):
        states += np.array([robot[variable] for robot in data[f'Robot {j}']])[1:] - 0.2
        
    states = states / ((r+1) * timesteps)

    plt.plot(timesteps, states, label=f'{r+1} Robots')

        
plt.title(f'Average Task Time as percentage of Total Time over Time (s)\n\nper # of Robots with 1 Health Station with 1 Bay')
plt.xlabel('Time (s)')
plt.ylabel('Task Time (%)')
plt.legend()
plt.show()


r = 9
for hs in range(0, num_health_stations):
    for bay in range(0, num_hs_bays):
        data = json.load(open(f'{r+1}_{hs+1}_{bay+1}.json', 'r'))

        timesteps = np.array([robot['timestep'] for robot in data[f'Robot 0']])[1:]
        states = np.zeros(timesteps.shape)

        # get treasures per robot over time
        for j in range(0, r + 1):
            states += np.array([robot[variable] for robot in data[f'Robot {j}']])[1:] - 0.2
            
        states = states / ((r+1) * timesteps)

        plt.plot(timesteps, states, label=f'{hs+1} HS, {bay+1} Bays')


plt.title(f'Average Task Time as percentage of Total Time over Time (s)\nper # of Health Stations and # of Bays with {r+1} Robots')
plt.xlabel('Time (s)')
plt.ylabel('Task Time (%)')
plt.legend()
plt.show()