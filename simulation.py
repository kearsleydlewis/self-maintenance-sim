import numpy as np
import matplotlib.pyplot as plt
import time
import signal
import threading
import random
import argparse
import sys
# import pandas as pd
from enum import Enum
import json

def get_shovel_degradation_fn(version):
    if version == '1.2':
        return lambda dig_time : dig_time*2.25
    elif version == '1.1':
        return lambda dig_time : dig_time*2.5
    else:
        return lambda dig_time : dig_time*3
    
def get_wheel_degradation_fn(version, faulty=False):
    if faulty:
        rand = np.random.randint(3,16)
    else:
        rand = np.random.randint(1,8)

    if version == '1.2':
        return lambda distance : distance*0.1*rand
    elif version == '1.1':
        return lambda distance : distance*0.2*rand
    else:
        return lambda distance : distance*0.3*rand
    
class PeriodicSleeper(threading.Thread):
    def __init__(self, task_function, period):
        super().__init__()
        self.end = False
        self.task_function = task_function
        self.period = period

        self.i = 0
        self.t0 = time.time()

    def sleep(self):
        self.i += 1
        delta = self.t0 + self.period*self.i - time.time()
        if (delta > 0):
            time.sleep(delta)
    
    def run(self):
        while True:
            if (self.end):
                break
            self.task_function()
            self.sleep()

    def stop(self):
        self.end = True

class STATUS(Enum):
    RED = 0
    YELLOW = 1
    GREEN = 2

class ROBOT_STATE(Enum):
    TASK = 0
    GO_TO_HS = 1
    HEALING = 2

class ROBOT_TASK(Enum):
    MOVE = 0
    DIG = 1

class Component:

    def __init__(self, name: str, version: str, degradation_fn):
        self.name = name
        self.version = version
        self.durability = 100.0
        self.status = STATUS.GREEN

        self.green_yellow = 25
        self.yellow_red = 10

        self.fixes = 0
        self.degradation_factor = 0.02

        self.degradation_fn = degradation_fn

    def set_status_thresholds(self, green_yellow=25, yellow_red=10):
        self.green_yellow = green_yellow
        self.yellow_red = yellow_red

    def degrade(self, value, broken=False):
        degredation_value = self.degradation_fn(value)
        if broken:
            self.durability = 0.0
        else:
            self.durability -= degredation_value * (1 + (self.fixes * self.degradation_factor))

        self.update_status()

    def heal(self, heal_value, full=False, half=False):
        if half:
            self.durability = 50.0
        elif full:
            self.durability = 100.0
        else:
            self.durability += heal_value
        
        self.fixes += 1

        self.update_status()
    
    def update_status(self):
        if (self.durability <= self.yellow_red):
            self.status = STATUS.RED
        elif (self.durability <= self.green_yellow):
            self.status = STATUS.YELLOW
        else:
            self.status = STATUS.GREEN

    def get_data(self):
        return {'name': self.name, 'version': self.version, 'durability': self.durability, 'status': self.status.value}

class Robot:
    def __init__(self,name, size: int, start_pos=np.array([0.0,0.0]), velocity: float = 1.0, real_time=True):

        self.name = name
        self.real_time = real_time

        # moving settings
        self.pos = start_pos
        self.vel = velocity # default 1 m/s
        self.size = size
        self.distance_traveled = 0.0

        # dig settings
        self.dig_time = 0
        self.treasures = 0

        # state settings
        self.state = ROBOT_STATE.TASK # task | go_to_hs | healing
        self.task_up_time = 0 # total time doing task
        self.task = ROBOT_TASK.MOVE # move | dig
        self.heal_next = False
        self.heal_now = False
        self.request_hs_location = False
        self.task_start_move()

        # components
        self.components = {
            'shovel': Component('shovel', '1.0', get_shovel_degradation_fn('1.0')),
            'left_wheel': Component('wheel', '1.0', get_wheel_degradation_fn('1.0', faulty=False)),
            'right_wheel': Component('wheel', '1.0', get_wheel_degradation_fn('1.0', faulty=True))
        }

        self.component_mapping = {
            ROBOT_TASK.MOVE: ['left_wheel', 'right_wheel'],
            ROBOT_TASK.DIG: ['shovel']
        }

        self.health_manager = RobotHealthManager(self, 2, real_time)
        self.hs_location = None
        self.assigned_hs = None
        self.at_hs = False
        self.in_queue = False

        if (self.real_time):
            self.health_manager.start()

    def step(self, dt):
        if (self.state == ROBOT_STATE.TASK):
            self.task_up_time += dt
            if (self.task == ROBOT_TASK.MOVE):
                self.move(dt, self.goal)
            else:
                self.dig(dt)

        if (self.state == ROBOT_STATE.GO_TO_HS):
            if self.hs_location is not None and not self.at_hs and not self.in_queue:
                self.move(dt, self.hs_location)

        if (self.state == ROBOT_STATE.HEALING):
            pass

        if not self.real_time:
            self.health_manager.step(dt)

    def move(self, dt, goal):
        velocity = self.vel

        # when both wheels are broken, we cannot move.
        # if one wheel is broken, move at half speed.
        for c in self.component_mapping[self.task]:
            if self.components[c].durability <= 0.0:
                velocity -= self.vel*0.5 

        # move toward goal
        v = (goal - self.pos) / np.linalg.norm(goal - self.pos)
        update = dt*(velocity*v)

        if (np.linalg.norm(goal - self.pos) < np.linalg.norm(update)):
            # goal reached
            update = goal - self.pos
            self.pos = goal
            if (self.state == ROBOT_STATE.TASK):
                self.task_start_dig()
            else:
                self.at_hs = True
        else:
            self.pos = self.pos + update
        
        for c in self.component_mapping[self.task]:
            self.components[c].degrade(np.linalg.norm(update))

        self.distance_traveled += np.linalg.norm(update)

    def dig(self, dt):
        for c in self.component_mapping[self.task]:
            if self.components[c].durability <= 0.0:
                # shovel is broken, cannot dig anymore, no treasure for you
                self.task_start_move()
                return

        # keep waiting for dig to finish
        self.dig_time -= dt
        if (self.dig_time <= 0):
            self.treasures += 1
            # finish dig
            self.task_start_move()

        for c in self.component_mapping[self.task]:
            self.components[c].degrade(dt)

    def task_start_dig(self):
        if self.heal_next:
            self.state = ROBOT_STATE.GO_TO_HS
        else:
            self.task = ROBOT_TASK.DIG
            self.dig_time = (float)(np.random.randint(3,7)) # random time between 3-7 seconds

    def task_start_move(self):
        if self.heal_next:
            self.state = ROBOT_STATE.GO_TO_HS
        else:
            self.task = ROBOT_TASK.MOVE
            self.goal = self.size*np.random.rand(2)
    

    def go_to_hs(self, now=False):
        if (self.state == ROBOT_STATE.TASK):
            if now:
                self.state = ROBOT_STATE.GO_TO_HS
            else:
                self.heal_next = True

            self.request_hs_location = True
            self.at_hs = False

    def set_hs_location(self, health_station):
        self.request_hs_location = False
        self.hs_location = health_station.pos
        self.assigned_hs = health_station

    def start_healing(self):
        self.state = ROBOT_STATE.HEALING
        self.in_queue = False

    def finish_healing(self):
        self.state = ROBOT_STATE.TASK
        self.heal_now = False
        self.heal_next = False
        self.health_manager.reset_prevent_timer()
        self.hs_location = None
        self.at_hs = False
        self.task_start_move()

    def get_data(self):
        return {'name': self.name, 'state': self.state.value, 'task_up_time': self.task_up_time, 'task': self.task.value, 'treasures': self.treasures, 'total_distance': self.distance_traveled, 'shovel': self.components['shovel'].get_data(), 'left_wheel': self.components['left_wheel'].get_data(), 'right_wheel': self.components['right_wheel'].get_data()}

class HealthStation:
    def __init__(self, name, start_pos, num_bays):
        self.name = name
        self.pos = start_pos
        self.num_bays = num_bays
        self.components = {
            'shovel': {'stock': [], 'order': [], 'order_time': 0 },
            'wheel':  {'stock': [], 'order': [], 'order_time': 0 },
        }

        self.heal_time = 10

        self.queue = []
        self.bays = {}

        for b in range(1, num_bays + 1):
            self.bays[b] = None # empty bay
            # bay is (time_left, robot)

    def step(self, dt):
        for key, bay in self.bays.items():
            if bay is not None:
                # subtract healing time
                bay[0] -= dt

                # healing finished, clean up
                if bay[0] <= 0.0:
                    bay[1].finish_healing()
                    self.bays[key] = None
            else:
                # bay is open
                if len(self.queue) > 0:
                    robot = self.queue.pop(0)
                    self.bays[key] = [self.heal_time, robot]
                    robot.start_healing()
                    self.start_healing(robot)

        
        for key, c in self.components.items():
            stock, order, order_time = c.values()
            if (order_time > 0):
                # subtract order time
                c['order_time'] = order_time - dt
                if (c['order_time'] <= 0):
                    # fulfull order if ready
                    stock.extend(order)
                    c['stock'] = stock
                    c['order'] = []
                    c['order_time'] = 0.0

            if (len(stock) < 3 and order_time <= 0):
                # create order
                new_order = []
                for i in range(0, np.random.randint(1,4)):
                    version = random.choice(['1.0', '1.1', '1.2'])

                    new_order.append(Component(key, version, get_shovel_degradation_fn(version) if key == 'shovel' else get_wheel_degradation_fn(version)))
                
                c['order'].extend(new_order)
                c['order_time'] = 20.0
                
    def start_healing(self, robot: Robot):
        # analyze robot, determine plan, heal
        for key, component in robot.components.items():
            if (component.fixes > 3 and len(self.components[component.name]['stock']) > 0):
                # replace if possible
                robot.components[key] = self.components[component.name]['stock'].pop(0)
            else:
                component.heal((100 - component.durability) - (component.fixes * 10))


    def restock_component(self, component: Component):
        self.components[component.name]['stock'].append(component)

    def restock_components(self, components):
        for c in components:
            self.restock_component(c)

    def add_robot_to_queue(self, robot):
        self.queue.append(robot)
    
    def getOpenBay(self):
        for key, bay in self.bays.items():
            if bay is None:
                # return first open bay
                return key

        # no bay open
        return None
    
    def get_data(self):
        return {'name': self.name, 'total_bays': self.num_bays, 'empty_bays': len([b for b in self.bays.values() if b is None]), 'queue_size': len(self.queue)}


class RobotHealthManager(threading.Thread):

    def __init__(self, robot: Robot, period, real_time):
        super().__init__()
        # robot settings
        self.robot = robot
        self.preventative_schedule = 300 # get checkup every 300 seconds
        self.preventative_schedule_t = self.preventative_schedule

        # threading settings
        self.end = False
        self.period = period
        self.i = 0
        self.daemon = True
        self.t0 = time.time()
        self.real_time = real_time

        self.current_time = 0.0

    def sleep(self):
        self.i += 1
        delta = self.t0 + self.period*self.i - time.time()
        if (delta > 0):
            time.sleep(delta)

    def step(self, dt):
        self.conduct_health_check()
        self.current_time += dt
    
    def run(self):
        while True:
            if (self.end):
                break
            self.conduct_health_check()
            self.sleep()

    def stop(self):
        self.end = True

    def conduct_health_check(self):
        for component in self.robot.components.values():
            if component.status == STATUS.RED:
                self.robot.go_to_hs(True)
            elif component.status == STATUS.YELLOW:
                self.robot.go_to_hs()
            else:
                pass
        
        if (self.preventative_schedule_t <= 0):
            self.robot.go_to_hs()
        else:
            self.preventative_schedule_t -= self.period
        
    def reset_prevent_timer(self):
        self.preventative_schedule_t = self.preventative_schedule

        
class HealthManager:

    def __init__(self) -> None:
        pass

class Simulation:

    def __init__(self, map_size, health_locs, num_robots, num_hs, num_bays, dt=0.1, max_time=200, real_time=True, data_output=False):
        self.map_size = map_size
        self.dt = dt
        self.max_time = max_time
        self.current_time = 0.0
        self.real_time = real_time
        
        self.data = {}
        self.filename = f'{num_robots}_{num_hs}_{num_bays}.json'
        self.data_output = data_output
        
        self.robots = [] # : list[Robot]
        for i in range(0, num_robots):
            robot = Robot(f'Robot {i}', map_size, start_pos=map_size*np.random.rand(2), real_time=real_time)
            self.robots.append(robot)
            self.data[robot.name] = []
        
        self.health_stations = []
        for j in range(0, num_hs):
            hs = HealthStation(f'HS {j}', health_locs[j], num_bays)
            self.health_stations.append(hs)
            self.data[hs.name] = []

        self.period = dt
        self.i = 0
        self.t0 = time.time()
        self.end = False

    def visualize(self):
        plt.cla()
        
        plt.xlim([0, int(self.map_size)])
        plt.ylim([0, int(self.map_size)])

        for robot in self.robots:
            plt.plot(robot.pos[0], robot.pos[1], '-bo')
            plt.plot(robot.goal[0], robot.goal[1], '-k.')

        for hs in self.health_stations:
            plt.plot(hs.pos[0], hs.pos[1], '-r+')

        plt.pause(0.0001)

    def main(self):
        #print(len(self.robots))
        for robot in self.robots:
            robot.step(self.dt)

            # check if robot is going to health station
            if robot.request_hs_location:
                # find closest hs
                closest = None
                distance = self.map_size*2
                for hs in self.health_stations:
                    dis = np.linalg.norm(hs.pos - robot.pos)
                    if (dis < distance):
                        closest = hs
                        distance = dis
                robot.set_hs_location(closest)

            # check if robot arrived at health station
            if robot.at_hs and not robot.in_queue and robot.state != ROBOT_STATE.HEALING:
                robot.assigned_hs.add_robot_to_queue(robot)
                robot.in_queue = True

        # step health station
        for hs in self.health_stations:
            hs.step(self.dt)

            h = hs.get_data()
            h['timestep'] = self.current_time
            self.data[hs.name].append(h)

        for robot in self.robots:
            d = robot.get_data()
            d['timestep'] = self.current_time
            self.data[robot.name].append(d)

        #self.print_stats()

        
        if (self.real_time):
            self.visualize()

        self.current_time += self.dt
        if (self.current_time >= self.max_time):
            # end simulation
            self.stop()

        data = []

    def print_stats(self):
        print(f'Bays: {self.health_station.bays}',
              f'Queue: {self.health_station.queue}',
              f'Components: {self.health_station.components}', sep='\n', end='\r')

    def start(self):
        self.fig = plt.figure(1, figsize=[7,7], dpi=140)

        while not self.end:
            if (self.end):
                break
           
            self.main()
            if (self.real_time):
                self.sleep()

    def sleep(self):
        self.i += 1
        delta = self.t0 + self.period*self.i - time.time()
        if (delta > 0):
            time.sleep(delta)

    def stop(self):
        self.end = True
        plt.cla()
        plt.close()

        for robot in self.robots:
            robot.health_manager.stop()

        if self.data_output:
            with open(self.filename, 'w', encoding='utf-8') as f:
                print(f'{self.filename} dump')
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        else:
            sys.exit(0)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='Self-Maintenance Simulation')

    parser.add_argument('-s', '--map_size', type=int, default=20, help="Size of the square map in meters (default = 20m).")
    parser.add_argument('-t', '--max_time', type=int, default=600, help="Maximum time of the simulation (default = 600s).")
    parser.add_argument('-dt', '--time_step', type=float, default=0.2, help="Size of simulation time step (default = 0.2s).")
    parser.add_argument('--real_time', help="Run the simulation in real time, which will show plot in real time (unless there are too many robots/hs, then it will lag a bit).", action="store_true")

    parser.add_argument('--data_output', help="Generate the data for plotting instead of running simulation", action="store_true")

    args = parser.parse_args()

    map_size = args.map_size
    max_time = args.max_time # 10 min
    dt = args.time_step
    real_time = args.real_time
    
    # health station positions for 1, 2, and 3 health stations (assumes size 20 map)
    # positions are arbitrary, but chosen to minimize distance to any health station
    # 1-3 hs
    hs_pos = {}
    hs_pos[1] = [np.array([0.5 * map_size, 0.5 * map_size])]
    hs_pos[2] = [np.array([0.333 * map_size, 0.333 * map_size]), np.array([0.666 * map_size, 0.666 * map_size])]
    hs_pos[3] = [np.array([0.25 * map_size, 0.75 * map_size]), np.array([0.4 * map_size, 0.25 * map_size]),  np.array([0.75 * map_size, 0.6 * map_size])]

    print(map_size, max_time, dt, real_time)

    if (args.data_output):
        # 1-10 robots
        for num_robot in range(1, 11):
            # 1-3 health stations
            for num_hs in range(1,4):
                # 1-3 bays at each health station
                for num_bays in range(1, 4):
                    print(f'Robots: {num_robot}\tHealth Stations: {num_hs}\tBays: {num_bays}')
                    np.random.seed(42) # for reproducability 
                    Simulation(map_size, hs_pos[num_hs], num_robots=num_robot, num_hs=num_hs, num_bays=num_bays, dt=dt, max_time=max_time, real_time=False, data_output=True).start()

    else:
        sim = Simulation(map_size, hs_pos[3], 10, 3, 3, dt=dt, max_time=max_time, real_time=real_time, data_output=False)

        signal.signal(signal.SIGINT, sim.stop)

        sim.start()