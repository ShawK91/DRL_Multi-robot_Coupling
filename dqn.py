import numpy as np, os, math
import mod_dqn as mod
from random import randint
import random
import cPickle
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
import numpy as np, torch
import torch.nn.functional as F
from scipy.special import expit
import fastrand, math
from torch.optim import Adam



def prob_choice(prob):
    prob = prob/np.sum(prob)
    rand = random.random()
    mass = 0.0
    for i, val in enumerate(prob):
        mass += val
        if rand <+ mass: return i

def soft_argmax(prob):
    return np.random.choice(np.flatnonzero(prob == prob.max()))

class Parameters:
    def __init__(self):

        #NN specifics
        self.num_hnodes = self.num_mem = 100

        # Train data
        self.batch_size = 10000
        self.num_episodes = 200000
        self.actor_epoch = 1; self.actor_lr = 0.005
        self.critic_epoch = 1; self.critic_lr = 0.005


        #Rover domain
        self.dim_x = self.dim_y = 10; self.obs_radius = 15; self.act_dist = 0.1; self.angle_res = 1
        self.num_poi = 5; self.num_rover = 1; self.num_timestep = 20
        self.poi_rand = 1

        #Dependents
        self.state_dim = 2*360 / self.angle_res #+ 2
        self.action_dim = 5
        self.epsilon = 0.5
        self.alpha = 0.9
        self.gamma = 0.9

        #Replay Buffer
        self.buffer_size = 1000000
        self.replay_buffer_choice = 1 #1: Normal Buffer (uniform sampling no prioritization)
                                      #2: Proportional sampling (priortizied)
                                      #3: Rank based (prioritized)
                                      #4: Open AI Proportional Experience Replay #TODO
        self.conf = {'batch_size': 32, 'bera_zero': 0.5, 'learn_start':1000, 'total_steps':self.num_episodes,
                     'size': 1000, 'replace_old':True, 'alpha':0.7}

        self.save_foldername = 'R_Block/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

        #unit tests
        self.unit_test = 0

class Task_Rovers:
    def __init__(self, parameters):
        self.params = parameters; self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y

        # Initialize food position container
        self.poi_pos = [[None, None] for _ in range(self.params.num_poi)]  # FORMAT: [item] = [x, y] coordinate
        self.poi_status = [False for _ in range(self.params.num_poi)]  # FORMAT: [item] = [T/F] is observed?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.params.num_rover)]  # Track each rover's position

    def reset_poi_pos(self):

        if self.params.unit_test == 1: #Unit_test
            self.poi_pos[0] = [0,1]
            return

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

        if self.params.unit_test == 1: #Unit test
            self.rover_pos[0] = [end,0];
            return

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
        #return mod.unsqueeze(np.array([self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]]), 1)

        self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]

        state = np.zeros(((360 / self.params.angle_res), 2))  # FORMAT: [bracket] = (drone_avg_dist, drone_number, food_avg_dist, food_number_item, reward ......]
        temp_poi_dist_list = [[] for _ in xrange(360 / self.params.angle_res)]
        temp_rover_dist_list = [[] for _ in xrange(360 / self.params.angle_res)]

        # Log all distance into brackets for POIs
        for loc, status in zip(self.poi_pos, self.poi_status):
            if status == True: continue #If accessed ignore

            x1 = loc[0] - self_x; x2 = -1.0
            y1 = loc[1] - self_y; y2 = 0.0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            if dist > self.params.obs_radius: continue #Observability radius

            bracket = int(angle / self.params.angle_res)
            temp_poi_dist_list[bracket].append(dist)

        # Log all distance into brackets for other drones
        for id, loc, in enumerate(self.rover_pos):
            if id == rover_id: continue #Ignore self

            x1 = loc[0] - self_x; x2 = -1.0
            y1 = loc[1] - self_y; y2 = 0.0
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            if dist > self.params.obs_radius: continue #Observability radius

            bracket = int(angle / self.params.angle_res)
            temp_rover_dist_list[bracket].append(dist)


        ####Encode the information onto the state
        for bracket in range(int(360 / self.params.angle_res)):
            # POIs
            num_poi = len(temp_poi_dist_list[bracket])
            if num_poi > 0: state[bracket][0] = sum(temp_poi_dist_list[bracket]) / num_poi
            else: state[bracket][0] = -1

            #Rovers
            num_rover = len(temp_rover_dist_list[bracket])
            if num_rover > 0: state[bracket][1] = sum(temp_rover_dist_list[bracket]) / num_rover
            else: state[bracket][1] = -1





        #state[-2,0], state[-1,0] = self_x, self_y
        #state = mod.unsqueeze(state.flatten(), 1)

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

    def step(self, rover_id, action):

        reward = 0.0

        if action == 0: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]]
        elif action == 1: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1] + 1]
        elif action == 2: new_pos = [self.rover_pos[rover_id][0] - 1, self.rover_pos[rover_id][1]]
        elif action == 3: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1] - 1]
        elif action == 4: new_pos = [self.rover_pos[rover_id][0] + 1, self.rover_pos[rover_id][1]]

        #Check if action is legal
        if not(new_pos[0] >= self.dim_x or new_pos[0] < 0 or new_pos[1] >= self.dim_y or new_pos[1] < 0):  #If legal
            self.rover_pos[rover_id] = [new_pos[0], new_pos[1]] #Execute action
        #else: reward -= 0.1 #Wall penalty

        #Check for reward
        reward = 0.0
        for i, loc in enumerate(self.poi_pos):
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
            dist = math.sqrt(x1 * x1 + y1 * y1)
            if dist <= self.params.act_dist:
                self.poi_status[i] = True
                reward += 1.0
            #reward -= dist/(2.0 * self.dim_x )

        #Try


        return reward/(1.0 * self.params.num_poi)

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
        if len(self.all_tracker[0][0]) > 100: #Assume all variable are updated uniformly
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

        for rover_id in range(parameters.num_rover):
            # Get current state from environment
            state = env.get_state(rover_id)
            state = mod.to_tensor(state)

            # Get action
            action_prob = agent.ac.actor_forward(state)
            action_prob = mod.to_numpy(action_prob).flatten()
            action = np.argmax(action_prob)

            # Run enviornment one step up and get reward
            reward = env.step(rover_id, action)
            episode_reward += reward


            #print action_prob, action
        env.visualize() #Visualize

    return episode_reward

def v_check(env, critic, params):
    env.reset()
    grid = [[None for _ in range(params.dim_x)] for _ in range(params.dim_y)]
    for i in range(params.dim_x):
        for j in range(params.dim_x):
            env.rover_pos[0] = [i,j]
            #env.poi_status[0] = False
            state = env.get_state(0)
            state = mod.to_tensor(state)
            val = mod.to_numpy(critic.critic_forward(state)).flatten()[0]
            grid[i][j] = "%0.2f" %val

    env.reset()
    for row in grid:
        print row
    print

def actor_check(env, actor, params):
    env.reset()
    symbol = ['S','R', 'U', 'L', 'D']
    grid = [[None for _ in range(params.dim_x)] for _ in range(params.dim_y)]
    for i in range(params.dim_x):
        for j in range(params.dim_x):
            env.rover_pos[0] = [i,j]
            #env.poi_status[0] = False
            state = env.get_state(0)
            state = mod.to_tensor(state)
            action = mod.to_numpy(actor.actor_forward(state)).flatten()
            action = np.argmax(action)
            grid[i][j] = symbol[action]

    env.reset()
    for row in grid:
        print row
    print

def add_experience(args, state, new_state, action, reward, agent):
    if args.replay_buffer_choice == 1: agent.replay_buffer.add(state, new_state, action, reward)
    if args.replay_buffer_choice == 2 or args.replay_buffer_choice == 3: agent.replay_buffer.store([state, new_state, action, reward])


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    env = Task_Rovers(parameters)
    agent = mod.A2C_Discrete(parameters)
    actor_noise = mod.OrnsteinUhlenbeckActionNoise(mu=np.zeros(parameters.action_dim))
    tracker = Tracker(parameters, ['rewards'], '')

    explore_success = 0.0; oracle = [3,3,3,3,2,2,2,0,0,0]
    oracle = [2,2,2,2,2,2,2,2,2,2]

    for episode in range(1, parameters.num_episodes): #Each episode
        episode_reward = 0.0

        env.reset() #Reset environment
        for timestep in range(parameters.num_timestep): #Each timestep

            for rover_id in range(parameters.num_rover):
                #Get current state from environment
                state = env.get_state(rover_id)
                state = mod.to_tensor(state)
                state = state.unsqueeze(0).unsqueeze(0)

                #Get action
                action_prob = agent.ac.actor_forward(state)
                action_prob = mod.to_numpy(action_prob).flatten()

                #Epsilon greedy exploration
                if random.random() < parameters.epsilon and episode % 10 != 0: #Explore
                    #action = prob_choice(action_prob)
                    action = randint(0,4)
                    #action = oracle[timestep]
                else:
                    action = soft_argmax(action_prob)
                #print action_prob

                #Run enviornment one step up and get reward
                reward = env.step(rover_id, action)
                episode_reward += reward

                #Get new state
                new_state = mod.to_tensor(env.get_state(rover_id))
                new_state = new_state.unsqueeze(0).unsqueeze(0)

                #Add to memory
                add_experience(parameters, state, new_state, action, reward, agent)

        if episode_reward > 0: explore_success += 1

        #Gradient update periodically
        #print episode, episode_reward
        if episode % 20 == 0:
            tracker.update([episode_reward], episode)
            agent.update_critic(episode)
            if episode % 20 == 0:
                agent.update_actor(episode)
                #parameters.epsilon *= 0.99 #Anneal epsilon
            print 'Gen', episode, 'Reward', episode_reward, 'Aggregrate', "%0.2f" % tracker.all_tracker[0][1], 'Exp_Success:', "%0.2f" % (explore_success/episode), 'Epsilon', "%0.2f" %parameters.epsilon#, 'Mem_size', agent.replay_buffer.size()


        #if episode % 200 == 0: test_env(env, agent, parameters)
        #if episode % 50 == 0: v_check(env, agent.ac, parameters)
        #if episode % 50 == 0: actor_check(env, agent.ac, parameters)





















