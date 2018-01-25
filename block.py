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
        self.num_hnodes = self.num_mem = 10

        # Train data
        self.batch_size = 1000
        self.num_episodes = 10000
        self.discount = 0.9
        self.actor_epoch = 5; self.actor_lr = 0.001
        self.critic_epoch = 10; self.critic_lr = 0.005


        #Rover domain
        self.dim_x = self.dim_y = 10; self.obs_radius = 10.0; self.act_dist = 2.0; self.angle_res = 5
        self.num_poi = 1; self.num_rover = 1; self.num_timestep = 10
        self.poi_rand = False

        #Dependents
        self.state_dim = 720 / self.angle_res
        self.action_dim = 2
        self.epsilon = 1.0


        self.save_foldername = 'R_Word/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

class Task_Rovers:
    def __init__(self, parameters):
        self.params = parameters; self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y

        # Initialize food position container
        self.poi_pos = [[None, None] for _ in range(self.params.num_poi)]  # FORMAT: [item] = [x, y] coordinate
        self.poi_status = [False for _ in range(self.params.num_poi)]  # FORMAT: [item] = [T/F] is observed?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.params.num_rover)]  # Track each rover's position

    def reset_poi_pos(self):
        start = 1.0;
        end = self.dim_x - 1.0
        rad = int(self.dim_x / math.sqrt(3) / 2.0)
        center = int((start + end) / 2.0)

        if self.params.poi_rand: #Random
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

        else: #Not_random
            for i in range(parameters.num_poi):
                if i % 3 == 0:
                    x = start + i/4 #randint(start, center - rad - 1)
                    y = start + i/3
                elif i % 3 == 1:
                    x = center + i/4 #randint(center + rad + 1, end)
                    y = start + i/4#randint(start, end)
                elif i % 3 == 2:
                    x = center+i/4#randint(center - rad, center + rad)
                    y = start + i/4#randint(start, center - rad - 1)
                else:
                    x = center+i/4#randint(center - rad, center + rad)
                    y = center+i/4#randint(center + rad + 1, end)
                self.poi_pos[i] = [x, y]

    def reset_poi_status(self):
        self.poi_status = self.poi_status = [False for _ in range(self.params.num_poi)]

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
        self.reset_poi_status()
        self.reset_poi_pos()
        self.reset_rover_pos()

    def get_state(self, rover_id):
        self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]

        state = np.zeros(((360 / self.params.angle_res), 2))  # FORMAT: [bracket] = (drone_avg_dist, drone_number, food_avg_dist, food_number_item, reward ......]
        temp_poi_dist_list = [[] for _ in xrange(360 / self.params.angle_res)]

        # Log all distance into brackets for food
        for loc, status in zip(self.poi_pos, self.poi_status):
            if status == True: continue #If accessed ignore

            x1 = loc[0] - self_x; x2 = -1.0
            y1 = loc[1] - self_y; y2 = 0.0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            bracket = int(angle / self.params.angle_res)
            temp_poi_dist_list[bracket].append(dist)

        ####Encode the information onto the state
        for bracket in range(int(360 / self.params.angle_res)):
            # Drones
            state[bracket][1] = len(temp_poi_dist_list[bracket])
            if state[bracket][1] > 0:
                state[bracket][0] = sum(temp_poi_dist_list[bracket]) / len(temp_poi_dist_list[bracket])
            else:
                state[bracket][0] = self.dim_x + self.dim_y

        state = mod.unsqueeze(state.flatten(), 1)
        return state

    def get_angle_dist(self, x1, y1, x2,y2):  # Computes angles and distance between two predators relative to (1,0) vector (x-axis)
        dot = x2 * x1 + y2 * y1  # dot product
        det = x2 * y1 - y2 * x1  # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        angle = math.degrees(angle) + 180.0 + 270.0
        angle = angle % 360
        dist = x1 * x1 + y1 * y1
        dist = math.sqrt(dist)
        return angle, dist

    #Sequential state
    def bck_get_state(self, rover_id):  # Returns a flattened array around the rovers position
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
        new_pos = [self.rover_pos[rover_id][0]+ action[0], self.rover_pos[rover_id][1]+ action[1]]

        #Check if action is legal
        if not(new_pos[0] >= self.dim_x or new_pos[0] < 0 or new_pos[1] >= self.dim_y or new_pos[1] < 0):  #If legal
            self.rover_pos[rover_id] = [new_pos[0], new_pos[1]] #Execute action (teleport)

        #Check for reward
        reward = 1.0
        for i, loc in enumerate(self.poi_pos):
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
            dist = math.sqrt(x1 * x1 + y1 * y1)
            # if dist <= self.params.act_dist:
            #     self.poi_status[i] = True
            #     reward += 1.0
            reward -= dist/(2.0 * self.dim_x )

        #Try


        return reward/(1.0 * self.params.num_poi)

    #Teleport step
    def bck_step(self, rover_id, action):

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
        for rover_pos, symbol in zip(self.rover_pos, drone_symbol_bank):
            x = int(rover_pos[0]); y = int(rover_pos[1])
            print x,y
            grid[x][y] = symbol

        symbol_bank = ['Q', 'W', 'E', 'R', 'T', 'Y']
        poison_symbol_bank = ['1', "2", '3', '4', '5', '6']
        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]); y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

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


def test_env(env, agent, parameters):
    env.reset()  # Reset environment
    episode_reward = 0.0
    for timestep in range(parameters.num_timestep):  # Each timestep

        # Get current state from environment
        state = env.get_state(0)
        state = mod.to_tensor(state)

        # Get action
        action = agent.ac.actor_forward(state)
        action = mod.to_numpy(action).flatten()
        env.visualize()

        # Run enviornment one step up and get reward
        # reward = env.step(0, mod.to_numpy(action).flatten())
        reward = env.step(0, action.flatten())
        episode_reward += reward
        print 'Action:', action.flatten(), 'Reward:', episode_reward, 'Pos:', env.rover_pos[0]




if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    env = Task_Rovers(parameters)
    agent = mod.DDPG(parameters)
    actor_noise = mod.OrnsteinUhlenbeckActionNoise(mu=np.zeros(parameters.action_dim))

    for episode in range(1, parameters.num_episodes): #Each episode
        episode_reward = 0.0


        env.reset() #Reset environment
        for timestep in range(parameters.num_timestep): #Each timestep

            #Get current state from environment
            state = env.get_state(0)
            state = mod.to_tensor(state)

            #Get action
            action = agent.ac.actor_forward(state)
            action = mod.to_numpy(action).flatten()

            if random.random() < parameters.epsilon and episode % 25 != 0: #Explore
                #print action,
                #action = action + actor_noise()
                action = np.array([random.random() - 0.5, random.random() - 0.5]) + action
                #action = np.array([-1.0, 1.0]) #TODO HACK
                #print action
            # else:
            #     #Exploratory action
            #     actor_noise = OrnsteinUhlenbeckProcess(mu=np.zeros(action_dim))
            #     action = np.array([2*(random.random()-0.5), 2*(random.random()-0.5)])

            #Run enviornment one step up and get reward
            reward = env.step(0, action.flatten())

            episode_reward += reward

            #Get new state
            new_state = env.get_state(0)

            #Add to memory
            agent.replay_buffer.add(mod.to_numpy(state), action, np.array([reward]), new_state)

        #Gradient update periodically
        #print episode, episode_reward
        if episode % 10 == 0:
            agent.update_critic()
            if episode % 20 == 0:
                agent.update_actor()
                parameters.epsilon *= 0.99 #Anneal epsilon
            print 'ACTUAL', episode, episode_reward/(parameters.num_timestep) ,parameters.epsilon
        print episode_reward/parameters.num_timestep

        if episode % 500 == 0: test_env(env, agent, parameters)





















