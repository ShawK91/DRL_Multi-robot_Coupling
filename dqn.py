import os, random, math
import mod_dqn as mod
from random import randint
import numpy as np, torch


def oracle2(env, timestep):
    if timestep <= 3: return [1,3]
    if env.poi_pos[0][0] == 4 and env.poi_pos[0][1] == 9: return [6,5]
    elif env.poi_pos[0][0] == 4 and env.poi_pos[0][1] == 0: return [5,6]




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
        self.num_episodes = 500000
        self.actor_epoch = 1; self.actor_lr = 0.005
        self.critic_epoch = 1; self.critic_lr = 0.005

        #Rover domain
        self.dim_x = self.dim_y = 10; self.obs_radius = 3; self.act_dist = 1.0; self.angle_res = 30
        self.num_poi = 2; self.num_rover = 6; self.num_timestep = 15
        self.poi_rand = 1
        self.coupling = 3
        self.rover_speed = 1
        self.sensor_model = 2 #1: Density Sensor
                              #2: Closest Sensor

        #Dependents
        self.state_dim = 2*360 / self.angle_res + 5
        self.action_dim = 7
        self.epsilon = 0.5; self.alpha = 0.9; self.gamma = 0.99
        self.reward_crush = 0.1 #Crush reward to prevent numerical issues

        #Replay Buffer
        self.buffer_size = 100000
        self.replay_buffer_choice = 1 #1: Normal Buffer (uniform sampling no prioritization)
                                      #2: Proportional sampling (priortizied)
                                      #3: Rank based (prioritized)
                                      #4: Open AI Proportional Experience Replay self.params.rover_speed0#TODO
        self.conf = {'batch_size': 1000, 'bera_zero': 0.5, 'learn_start':1000, 'total_steps':self.num_episodes,
                     'size': self.buffer_size, 'replace_old':True, 'alpha':0.7}

        self.save_foldername = 'R_Block/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)

        #Unit tests (Simply changes the rover/poi init locations)
        self.unit_test = 0 #0: None
                           #1: Single Agent
                           #2: Multiagent 2-coupled

class Task_Rovers:
    def __init__(self, parameters):
        self.params = parameters; self.dim_x = parameters.dim_x; self.dim_y = parameters.dim_y

        # Initialize food position container
        self.poi_pos = [[None, None] for _ in range(self.params.num_poi)]  # FORMAT: [item] = [x, y] coordinate
        self.poi_status = [False for _ in range(self.params.num_poi)]  # FORMAT: [item] = [T/F] is observed?

        # Initialize rover position container
        self.rover_pos = [[0.0, 0.0] for _ in range(self.params.num_rover)]  # Track each rover's position
        self.ledger_closest = [[0.0, 0.0] for _ in range(self.params.num_rover)]  # Track each rover's ledger call

        #Macro Action trackers
        self.util_macro = [[False, False, False] for _ in range(self.params.num_rover)] #Macro utilities to track [Is_currently_active?, Is_activated_now?, Is_reached_destination?]



    def reset_poi_pos(self):

        if self.params.unit_test == 1: #Unit_test
            self.poi_pos[0] = [0,1]
            return

        if self.params.unit_test == 2: #Unit_test
            if random.random()<0.5: self.poi_pos[0] = [4,0]
            else: self.poi_pos[0] = [4,9]
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
                    x = center + (rover_id / (4 * center - rad)) % (center - rad)-1
                    y = center - 1 + (rover_id / 4) % (center - rad)
                if quadrant == 2:
                    x = center + 1 + (rover_id / 4) % (center - rad)
                    y = center + (rover_id / (4 * center - rad)) % (center - rad)
                if quadrant == 3:
                    x = center - (rover_id / (4 * center - rad)) % (center - rad)
                    y = center + 1 - (rover_id / 4) % (center - rad)
                self.rover_pos[rover_id] = [x, y]

    def reset(self):
        self.reset_poi_pos()
        self.reset_rover_pos()
        self.poi_status = self.poi_status = [False for _ in range(self.params.num_poi)]
        self.util_macro = [[False, False, False] for _ in range(self.params.num_rover)]  # Macro utilities to track [Is_currently_active?, Is_activated_now?, Is_reached_destination?]

    def get_state(self, rover_id, ledger):
        if self.util_macro[rover_id][0]: #If currently active
            if self.util_macro[rover_id][1] == False: #Not first time activate (Not Is-activated-now)
                return np.zeros((720/self.params.angle_res + 5, 1)) -10000 #If macro return none
            else:
                self.util_macro[rover_id][1] = False  # Turn off is_activated_now?

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
            if num_poi > 0:
                if parameters.sensor_model == 1: state[bracket][0] = sum(temp_poi_dist_list[bracket]) / num_poi #Density Sensor
                else: state[bracket][0] = min(temp_poi_dist_list[bracket])  #Minimum Sensor
            else: state[bracket][0] = -1

            #Rovers
            num_rover = len(temp_rover_dist_list[bracket])
            if num_rover > 0:
                if parameters.sensor_model == 1: state[bracket][1] = sum(temp_rover_dist_list[bracket]) / num_rover #Density Sensor
                else: state[bracket][1] = min(temp_rover_dist_list[bracket]) #Minimum Sensor
            else: state[bracket][1] = -1

        state = state.flatten()
        #Append wall info
        state = np.concatenate((state, np.array([-1.0, -1.0, -1.0, -1.0])))
        if self_x <= self.params.obs_radius: state[-4] = self_x
        if self.params.dim_x - self_x <= self.params.obs_radius: state[-3] = self.params.dim_x - self_x
        if self_y <= self.params.obs_radius :state[-2] = self_y
        if self.params.dim_y - self_y <= self.params.obs_radius: state[-1] = self.params.dim_y - self_y

        #Add ledger and state vars
        closest_loc, min_dist = ledger.get_knn([self_x, self_y])
        if closest_loc[0] == None: closest_loc[0], closest_loc[1] = self_x, self_y
        self.ledger_closest[rover_id] = [closest_loc[0], closest_loc[1]]

        #if rover_id == 1: state =  np.zeros((720 / self.params.angle_res)) - 2 #TODO TEST
        state = np.concatenate((state, np.array([min_dist])))
        state = mod.unsqueeze(state, 1)
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

    def step(self, joint_action, ledger):

        for rover_id in range(parameters.num_rover):
            action = joint_action[rover_id]
            new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]] #Default position

            if action == 1: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1] + self.params.rover_speed]
            elif action == 2: new_pos = [self.rover_pos[rover_id][0] - self.params.rover_speed, self.rover_pos[rover_id][1]]
            elif action == 3: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1] - self.params.rover_speed]
            elif action == 4: new_pos = [self.rover_pos[rover_id][0] + self.params.rover_speed, self.rover_pos[rover_id][1]]

            elif action == 5: #Macro Action - Move towards the closest nbr in ledger
                tow_vector = [self.ledger_closest[rover_id][0] - self.rover_pos[rover_id][0], self.ledger_closest[rover_id][1] - self.rover_pos[rover_id][1]]
                if tow_vector[0] > 0: new_pos = [self.rover_pos[rover_id][0] + self.params.rover_speed, self.rover_pos[rover_id][1]]
                elif tow_vector[0] < 0: new_pos = [self.rover_pos[rover_id][0] - self.params.rover_speed, self.rover_pos[rover_id][1]]
                elif tow_vector[1] > 0: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]+self.params.rover_speed]
                elif tow_vector[1] < 0: new_pos = [self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]-self.params.rover_speed]

                if new_pos[0] == self.ledger_closest[rover_id][0] and new_pos[1] == self.ledger_closest[rover_id][1]:
                    self.util_macro[rover_id][0] = False #Turn off is active
                    self.util_macro[rover_id][2] = True  #Turn on Is_reached_destination

            elif action == 6:
                ledger.post([[self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]]], cost=1.0)

            #Check if action is legal
            if not(new_pos[0] >= self.dim_x or new_pos[0] < 0 or new_pos[1] >= self.dim_y or new_pos[1] < 0):  #If legal
                self.rover_pos[rover_id] = [new_pos[0], new_pos[1]] #Execute action

    def get_reward(self):
        #Update POI's visibility
        poi_visitors = [[] for _ in range(self.params.num_poi)]
        for i, loc in enumerate(self.poi_pos): #For all POIs
            if self.poi_status[i]== True: continue #Ignore POIs that have been harvested already

            for rover_id in range(self.params.num_rover): #For each rover
                x1 = loc[0] - self.rover_pos[rover_id][0]; y1 = loc[1] - self.rover_pos[rover_id][1]
                dist = math.sqrt(x1 * x1 + y1 * y1)
                if dist <= self.params.act_dist: poi_visitors[i].append(rover_id) #Add rover to POI's visitor list

        #Compute reward
        rewards = [0.0 for _ in range(self.params.num_rover)]
        for poi_id, rovers in enumerate(poi_visitors):
            if len(rovers) >= self.params.coupling:
                self.poi_status[poi_id] = True
                lucky_rovers = random.sample(rovers, self.params.coupling)
                for rover_id in lucky_rovers: rewards[rover_id] += 1.0 * self.params.reward_crush/self.params.num_poi


        return rewards

    def visualize(self):

        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        # Draw in hive
        drone_symbol_bank = ["0", "1", '2', '3', '4', '5']
        for rover_pos, symbol in zip(self.rover_pos, drone_symbol_bank):
            x = int(rover_pos[0]); y = int(rover_pos[1])
            #print x,y
            grid[x][y] = symbol


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


def visualize_episode(env, agent, parameters):
    episode_reward = 0.0
    env.reset()  # Reset environment
    agent.ledger.reset()  # Reset ledger
    for timestep in range(parameters.num_timestep):  # Each timestep

        # Get current state from environment
        joint_state = []
        for rover_id in range(parameters.num_rover): joint_state.append(
            mod.to_tensor(env.get_state(rover_id, agent.ledger)))
        joint_state_T = torch.cat(joint_state, 1)

        # Get action
        joint_action_prob = mod.to_numpy(agent.ac.actor_forward(joint_state_T))  # [probs, batch]
        #actions = np.argmax(joint_action_prob, axis=0)  # Greedy max value action selection
        greedy_actions = []  # Greedy actions breaking ties
        for i in range(len(joint_action_prob[0])):
            max = np.max(joint_action_prob[:, i])
            greedy_actions.append(np.random.choice(np.where(max == joint_action_prob[:, i])[0]))# greedy_actions = np.random.choice(np.flatnonzero(prob == prob.max())) #TODO Break Argmax bias towards index clashes (use random choice between same values)

        actions = np.array(greedy_actions)

        # Run enviornment one step up and get reward
        env.step(actions, agent.ledger)
        joint_rewards = env.get_reward()
        episode_reward += sum(joint_rewards) / parameters.coupling

        env.visualize()
    print agent.ledger.ledger

    return episode_reward

def trace_viz(env, agent, parameters):
    episode_reward = 0.0
    env.reset()  # Reset environment
    agent.ledger.reset()  # Reset ledger
    macro_experience = [[None, None, None, None] for _ in range(parameters.num_rover)]  # Bucket to store macro actions (time extended) experiences.
    rover_path = [[(loc[0], loc[1])] for loc in env.rover_pos]
    action_diversity = [[0, 0, 0, 0, 0,0,0] for _ in range(parameters.num_rover)]

    for timestep in range(1, parameters.num_timestep+1): #Each timestep

        # Get current state from environment
        joint_state = []
        for rover_id in range(parameters.num_rover): joint_state.append(mod.to_tensor(env.get_state(rover_id, agent.ledger)))
        joint_state_T = torch.cat(joint_state, 1)

        # Get action
        joint_action_prob = mod.to_numpy(agent.ac.actor_forward(joint_state_T)) #[probs, batch]

        greedy_actions = [] #Greedy actions breaking ties
        for i in range(len(joint_action_prob[0])):
            max = np.max(joint_action_prob[:,i])
            greedy_actions.append(np.random.choice(np.where(max == joint_action_prob[:,i])[0]))
        actions = np.array(greedy_actions)

        #Track actions
        for rover_id, a in enumerate(actions):
            action_diversity[rover_id][a] += 1
            if env.util_macro[rover_id][0]: action_diversity[rover_id][a] -= 1


        #Macro action to macro (time-extended macro action)
        for rover_id, entry in enumerate(env.util_macro):
             if actions[rover_id] == 5 and entry[0] == False: #Macro action first state
                 env.util_macro[rover_id][0] = True #Turn on is_currently_active?
                 env.util_macro[rover_id][1] = True #Turn on is_activated_now?
                 macro_experience[rover_id][0] = joint_state[rover_id]
                 macro_experience[rover_id][2] = actions[rover_id]

             if entry[0]:
                 actions[rover_id] = 5 #Macro action continuing

        # Run enviornment one step up and get reward
        env.step(actions, agent.ledger)
        joint_rewards = env.get_reward()
        episode_reward += sum(joint_rewards)/parameters.coupling

        #Process macro experiences and add to memory
        for rover_id, exp in enumerate(macro_experience):
            if env.util_macro[rover_id][2]: #If reached destination
                env.util_macro[rover_id][2] = False
                macro_experience[rover_id] = [None, None, None, None]

        #Append rover path
        for rover_id in range(parameters.num_rover): rover_path[rover_id].append((env.rover_pos[rover_id][0],env.rover_pos[rover_id][1]))


    #Visualize
    grid = [['-' for _ in range(env.dim_x)] for _ in range(env.dim_y)]

    drone_symbol_bank = ["0", "1", '2', '3', '4', '5', '6','7','8','9','10','11']
    # Draw in rover path
    for rover_id in range(parameters.num_rover):
        for time in range(parameters.num_timestep):
            x = int(rover_path[rover_id][time][0]);
            y = int(rover_path[rover_id][time][1])
            # print x,y
            grid[x][y] = drone_symbol_bank[rover_id]

    # Draw in food
    for loc, status in zip(env.poi_pos, env.poi_status):
        x = int(loc[0]);
        y = int(loc[1])
        marker = 'I' if status else 'A'
        grid[x][y] = marker

    for row in grid:
        print row
    print
    print action_diversity
    print agent.ledger.ledger
    print '------------------------------------------------------------------------'
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
    #actor_noise = mod.OrnsteinUhlenbeckActionNoise(mu=np.zeros(parameters.action_dim))
    tracker = Tracker(parameters, ['rewards'], '')



    for episode in range(1, parameters.num_episodes, 1): #Each episode
        episode_reward = 0.0; env.reset() #Reset environment
        agent.ledger.reset() #Reset ledger
        macro_experience = [[None, None, None, None] for _ in range(parameters.num_rover)] #Bucket to store macro actions (time extended) experiences.
        for timestep in range(1, parameters.num_timestep+1): #Each timestep

            # Get current state from environment
            joint_state = []
            for rover_id in range(parameters.num_rover): joint_state.append(mod.to_tensor(env.get_state(rover_id, agent.ledger)))
            joint_state_T = torch.cat(joint_state, 1)

            # Get action
            joint_action_prob = mod.to_numpy(agent.ac.actor_forward(joint_state_T)) #[probs, batch]

            greedy_actions = [] #Greedy actions breaking ties
            for i in range(len(joint_action_prob[0])):
                max = np.max(joint_action_prob[:,i])
                greedy_actions.append(np.random.choice(np.where(max == joint_action_prob[:,i])[0]))
            greedy_actions = np.array(greedy_actions)

            #Epsilon greedy exploration through action choice perturbation
            rand = np.random.uniform(0,1,parameters.num_rover)
            is_perturb = rand < parameters.epsilon
            if episode % 20 == 0: is_perturb = np.zeros(parameters.num_rover).astype(bool) #Greedy for these test episodes
            actions = np.multiply(greedy_actions, (np.invert(is_perturb))) + np.multiply(np.random.randint(0, parameters.action_dim, parameters.num_rover), (is_perturb))


            #ORACLE
            #if episode % 13 == 0: actions = oracle2(env, timestep)


            #Macro action to macro (time-extended macro action)
            for rover_id, entry in enumerate(env.util_macro):
                 if actions[rover_id] == 5 and entry[0] == False: #Macro action first state
                     env.util_macro[rover_id][0] = True #Turn on is_currently_active?
                     env.util_macro[rover_id][1] = True #Turn on is_activated_now?
                     macro_experience[rover_id][0] = joint_state[rover_id]
                     macro_experience[rover_id][2] = actions[rover_id]


                 if entry[0]: actions[rover_id] = 5 #Macro action continuing

            # Run enviornment one step up and get reward
            env.step(actions, agent.ledger)
            joint_rewards = env.get_reward()
            episode_reward += (sum(joint_rewards)/parameters.coupling)/parameters.reward_crush


            #Get new state
            joint_next_state = []
            for rover_id in range(parameters.num_rover): joint_next_state.append(mod.to_tensor(env.get_state(rover_id, agent.ledger)))

            #Add new state and reward to macro-action (time extended) considerations
            for rover_id, entry in enumerate(env.util_macro):
                if entry[2]: #If reached destination
                    macro_experience[rover_id][1] = joint_next_state[rover_id]
                    macro_experience[rover_id][3] = joint_rewards[rover_id]

            #Add to memory
            for state, new_state, action, reward in zip(joint_state, joint_next_state, actions, joint_rewards):
                if action == 5: continue #Skip the ones currently executing a macro action (not the one who just chose it).
                add_experience(parameters, state, new_state, action, reward, agent)

            #Process macro experiences and add to memory
            for rover_id, exp in enumerate(macro_experience):
                if env.util_macro[rover_id][2]: #If reached destination
                    env.util_macro[rover_id][2] = False
                    add_experience(parameters, macro_experience[rover_id][0], macro_experience[rover_id][1], macro_experience[rover_id][2], macro_experience[rover_id][3], agent)
                    macro_experience[rover_id] = [None, None, None, None]


        #Gradient update periodically
        if episode % 20 == 0:
            tracker.update([episode_reward], episode)
            agent.update_critic(episode)
            if episode % 20 == 0:
                agent.update_actor(episode)
                #parameters.epsilon *= 0.99 #Anneal epsilon
            print 'Gen', episode, 'Reward', episode_reward, 'Aggregrate', "%0.2f" % tracker.all_tracker[0][1], 'Epsilon', "%0.2f" %parameters.epsilon#, 'Mem_size', agent.replay_buffer.size() 'Exp_Success:', "%0.2f" % (explore_success/episode),


        #if episode % 200 == 0: visualize_episode(env, agent, parameters)
        if episode % 20 == 0: trace_viz(env, agent, parameters)
        #if episode % 50 == 0: v_check(env, agent.ac, parameters)
        #if episode % 50 == 0: actor_check(env, agent.ac, parameters)





















