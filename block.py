import numpy as np, os, math
import mod_block as mod
from random import randint
from torch.autograd import Variable
import torch
from torch.utils import data as util
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import random



class Parameters:
    def __init__(self):

        #NN specifics
        self.state_dim = 3
        self.action_dim = 2
        self.num_hnodes = 50
        self.num_mem = 50

        # Train data
        self.batch_size = 1000
        self.num_episodes = 1000000
        self.discount = 0.9

        #Rover domain
        self.dim_x = self.dim_y = 20; self.obs_radius = 5.0; self.act_dist = 1.0
        self.num_poi = 5; self.num_rover = 1; self.num_timestep = 5



        self.save_foldername = 'R_Word/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)


class Task_Rovers:
    def __init__(self, parameters):
        self.params = parameters; self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y

        # Initialize food position container
        self.poi_pos = [[None, None] for _ in range(self.params.num_poi)]  # FORMAT: [item] = [x, y] coordinate
        self.poi_status = [[False] for _ in range(self.params.num_poi)]  # FORMAT: [item] = [T/F] is observed?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.params.num_rover)]  # Track each rover's position


    def reset_poi_pos(self):
        start = 1.0;
        end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        for i in range(parameters.num_poi):
            if i % 3 == 0:
                x = randint(start, center - rad - 1)
                y = randint(start, end)
            elif i % 3 == 1:
                x = randint(center + rad + 1, end)
                y = randint(start, end)
            elif i % 3 == 2:
                x = randint(center - rad, center + rad)
                y = randint(start, center - rad - 1)
            else:
                x = randint(center - rad, center + rad)
                y = randint(center + rad + 1, end)
            self.poi_pos[i] = [x, y]

    def reset_food_status(self):
        self.poi_status = self.poi_status = [[False] for _ in range(self.params.num_poi)]

    def reset_rover_pos(self):

        start = 1.0; end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        for rover_id in range(self.params.num_rover):
            quadrant = rover_id % 4
            if quadrant == 0:
                x = center - 1 - (rover_id / 4) % (center - rad)
                y = center - (rover_id / (4 * center - rad)) % (center - rad)
            if quadrant == 1:
                x = center + (rover_id / (4 * center - rad)) % (center - rad)
                y = center - 1 + (rover_id / 4) % (center - rad)
            if quadrant == 2:
                x = center + 1 + (rover_id / 4) % (center - rad)
                y = center + (rover_id / (4 * center - rad)) % (center - rad)
            if quadrant == 3:
                x = center - (rover_id / (4 * center - rad)) % (center - rad)
                y = center + 1 - (rover_id / 4) % (center - rad)
            self.rover_pos[rover_id] = [x, y]

    def reset(self):
        self.reset_food_status()
        self.reset_poi_pos()
        self.reset_rover_pos()

    def get_state(self, rover_id):  # Returns a flattened array around the rovers position
        self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]
        #state = [[self_x, self_y, 0]] #Start by counting yourself
        state = []
        #TODO Count yourself

        # Include all pois
        for loc, status in zip(self.poi_pos, self.poi_status):
            if status == True: continue #Ignore POIs that have been harvested already

            x1 = loc[0] - self_x; y1 = loc[1] - self_y
            dist = math.sqrt(x1 * x1 + y1 * y1)
            if dist <= self.params.obs_radius:
                state.append([x1, y1, 1])
                break #TODO no lsTM FOR NOW

        # # Include all rovers
        # for id, loc in enumerate(self.rover_pos):
        #     if id  == rover_id: continue #Count yourself always at the front (prevent double counting)
        #     x1 = loc[0] - self_x; y1 = loc[1] - self_y
        #     dist = math.sqrt(x1 * x1 + y1 * y1)
        #     if dist <= self.params.obs_radius: state.append([x1, y1, 0])

        #TODO
        if len(state) == 0: state = [[0,0,0]]
        return np.array(state).transpose()

    def step(self, rover_id, action):

        #Check if action is legal
        if not(action[0] > self.dim_x or action[0] < 0 or action[1] > self.dim_y or action[1] < 0):  #If legal
            self.rover_pos[rover_id] = [action[0], action[1]] #Execute action (teleport)

        #Check for reward
        reward = 0.0
        for i, loc in enumerate(self.poi_pos):
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
            dist = math.sqrt(x1 * x1 + y1 * y1)
            if dist <= self.params.act_dist:
                self.poi_status[i] = True
                reward += 1.0

        return reward

    def visualize(self):

        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        # Draw in hive
        drone_symbol_bank = ["@", '#', '$', '%', '&']
        for drone_pos, symbol in zip(self.hive_pos, drone_symbol_bank):
            x = int(drone_pos[0]);
            y = int(drone_pos[1])
            grid[x][y] = symbol

        symbol_bank = ['Q', 'W', 'E', 'R', 'T', 'Y']
        poison_symbol_bank = ['1', "2", '3', '4', '5', '6']
        # Draw in food
        for sku_id in range(self.num_foodskus):
            if self.food_poison_info[sku_id]:  # If poisionous
                symbol = poison_symbol_bank.pop(0)
            else:
                symbol = symbol_bank.pop(0)
            for item_id in range(self.num_food_items):
                x = int(self.food_list[sku_id][item_id][0]);
                y = int(self.food_list[sku_id][item_id][1]);
                grid[x][y] = symbol

        for row in grid:
            print row
        print


class Tracker(): #Tracker
    def __init__(self, parameters, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = parameters.save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)

    def update(self, updates, generation):
        for update, var in zip(updates, self.all_tracker):
            var[0].append(update)

        #Constrain size of convolution
        if len(self.all_tracker[0][0]) > 10: #Assume all variable are updated uniformly
            for var in self.all_tracker:
                var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            var[1] = sum(var[0])/float(len(var[0]))

        if generation % 10 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')



if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    env = Task_Rovers(parameters)
    agent = mod.DDPG(parameters)

    for episode in range(1, parameters.num_episodes): #Each episode
        episode_reward = 0.0

        env.reset() #Reset environment
        for timestep in range(parameters.num_timestep): #Each timestep

            #Get current state from environment
            state = env.get_state(0)
            state = mod.to_tensor(state)

            #Get action
            if random.random() < 0.2:
                action = agent.ac.actor_forward(state)
                action = mod.to_numpy(action).flatten()
                #print action
            else:
                #Exploratory action
                action = np.array([randint(0, parameters.dim_x),randint(0, parameters.dim_y)])

            #Run enviornment one step up and get reward
            #reward = env.step(0, mod.to_numpy(action).flatten())
            reward = env.step(0, action.flatten())

            episode_reward += reward

            #Get new state
            new_state = env.get_state(0)

            #Add to memory
            if episode % 10 == 0: continue
            agent.replay_buffer.add(mod.to_numpy(state), action, np.array([reward]), new_state)

        #Gradient update periodically
        #print episode, episode_reward
        if episode % 10 == 0:
            agent.update_policy()
            print 'ACTUAL', episode, episode_reward





















