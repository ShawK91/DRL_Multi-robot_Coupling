import cPickle
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import random
from random import randint
import numpy as np, torch
import torch.nn.functional as F
from scipy.special import expit
import fastrand, math
from torch.optim import Adam


class A2C_Discrete(object):
    def __init__(self, args, agent_id):

        self.args = args; self.id = agent_id

        # Create Actor and Critic Network
        self.ac = Actor_Critic(args.state_dim, args.action_dim, args)
        self.target_net = Actor_Critic(args.state_dim, args.action_dim, args)
        self.ledger = Ledger(max_entries=10000, decay_rate=0.9)

        #Optimizers
        self.actor_optim = Adam(self.ac.parameters(), lr=args.actor_lr)
        self.critic_optim = Adam(self.ac.parameters(), lr=args.critic_lr)

        # Create replay buffer
        self.replay_buffer = Memory(self.args.buffer_size)

        # Hyper-parameters
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.criterion = nn.SmoothL1Loss()

    def synchronize(self):
        # for param, target_param in zip(list(self.ac.parameters()), list(self.target_net.parameters())):
        #     param.data = target_param.data

        self.ac.w_critic1.data = self.target_net.w_critic1.data
        self.ac.w_critic2.data = self.target_net.w_critic2.data
        self.ac.w_critic3.data = self.target_net.w_critic3.data
        #self.ac.w_critic4.data = self.target_net.w_critic4.data
        #self.ac.w_critic5.data = self.target_net.w_critic5.data
        self.ac.w_critic6.data = self.target_net.w_critic6.data

    def sample_memory(self, episode):
        data = self.replay_buffer.sample(self.args.batch_size)
        states = torch.cat([ o[1][0] for o in data], 1)
        new_states = torch.cat([ o[1][1] for o in data], 1)
        actions = [ o[1][2] for o in data]
        rewards = to_tensor(np.array([ o[1][3] for o in data])).unsqueeze(0)
        return states, new_states, actions, rewards, data

    def compute_dpp(self, states):
        vals = self.ac.critic_forward(states)

        states = to_numpy(states)
        mid_index = 180 / self.args.angle_res
        coupling = self.args.coupling
        #dpp_sweep = [mid_index + i for i in range(int(-coupling/2), int(-coupling/2) + coupling, 1)]

        dpp_sweep = random.sample(range(360/self.args.angle_res), coupling+1)

        for i in dpp_sweep:
            states[i,:] += 2.0
            vals = torch.cat((vals, self.ac.critic_forward(to_tensor(states))), 0)

        return torch.max(vals, 0)[0].unsqueeze(0)

    def update_actor(self, episode, is_dpp):
        # Sample batch
        states, new_states, actions, rewards, data = self.sample_memory(episode)

        if len(states) == 0: return

        if is_dpp:
            vals = self.compute_dpp(states)
            new_vals = self.compute_dpp(new_states)

        else:
            vals = self.ac.critic_forward(states)
            new_vals = self.ac.critic_forward(new_states)


        action_logs = self.ac.actor_forward(states)

        dt = rewards + self.gamma * new_vals - vals
        dt = to_numpy(dt)
        #print np.max(dt), np.min(dt)

        #Update priorities
        for i in range(len(data)): self.replay_buffer.update(data[i][0], abs(dt[0][i]))

        dt = to_tensor(dt)
        alogs = []
        for i, action in enumerate(actions):
            alogs.append(action_logs[action, i])
        alogs = torch.cat(alogs).unsqueeze(0)


        for epoch in range(self.args.actor_epoch):
            policy_loss =  -(dt * alogs)
            policy_loss = policy_loss.mean()
            policy_loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(self.ac.parameters(), 10)
            self.actor_optim.step()
            self.ac.zero_grad()

    def update_critic(self, episode):
        # Sample batch
        states, new_states, actions, rewards, data = self.sample_memory(episode)

        if len(states) == 0: return


        for epoch in range(self.args.critic_epoch):

            vals = self.target_net.critic_forward(states);
            new_states.volatile = True
            new_vals = self.target_net.critic_forward(new_states)
            targets = rewards + self.gamma * new_vals

            #print 'Val', np.max(to_numpy(vals).flatten()), np.min(to_numpy(vals).flatten())
            #print 'Targets', np.max(to_numpy(targets).flatten()), np.min(to_numpy(targets).flatten())

            loss = self.criterion(vals, targets)
            loss.backward(retain_graph = True)
            torch.nn.utils.clip_grad_norm(self.target_net.parameters(), 10)
            self.critic_optim.step()
            self.target_net.zero_grad()
            new_states.volatile = False


class Ledger():
    def __init__(self, max_entries, decay_rate):
        self.max_entries = max_entries; self.decay_rate = decay_rate
        self.ledger = list()

    def post(self, loc, cost):
        if cost > random.random(): self.ledger.append((1.0, loc))

    def get_knn(self, loc): #TODO Make it knn (currently k = 1)
        min_dist = 1000; closest_loc = [None, None]
        for _, entries in self.ledger:
            dist = abs(loc[0]-entries[0][0]) + abs(loc[1]-entries[0][1])
            if dist < min_dist:
                min_dist = dist
                closest_loc = [entries[0][0], entries[0][1]]
        if min_dist == 1000: min_dist = -1.0
        return closest_loc, min_dist

    def reset(self):
        self.ledger = list()

    def decay(self):
        del_list = []
        for stay_prob, entries in self.ledger:
            if stay_prob <= random.random():
                del_list.append((stay_prob, entries))
                continue
            stay_prob *= self.decay_rate

        #Delete entries
        for item in del_list:
            self.ledger.remove(item)

class Actor_Critic(nn.Module):
    def __init__(self, num_input, num_actions, args):
        super(Actor_Critic, self).__init__()
        self.args = args

        #Actor
        #Primitive
        self.w_actor1 = Parameter(torch.rand(args.num_hnodes, num_input), requires_grad=1)
        self.w_actor2 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        self.w_actor3 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        #self.w_actor4 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        #self.w_actor5 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        self.w_actor6 = Parameter(torch.rand(num_actions, args.num_hnodes), requires_grad=1)


        #Comm module
        #self.w_actor_comm1 = Parameter(torch.rand(args.num_hnodes, num_input), requires_grad=1)
        #self.w_actor_comm2 = Parameter(torch.rand(2, args.num_hnodes+4), requires_grad=1)

        #Critic
        self.w_critic1 = Parameter(torch.rand(args.num_hnodes, num_input), requires_grad=1)
        self.w_critic2 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        self.w_critic3 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        #self.w_critic4 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        #self.w_critic5 = Parameter(torch.rand(args.num_hnodes, args.num_hnodes), requires_grad=1)
        self.w_critic6 = Parameter(torch.rand(1, args.num_hnodes), requires_grad=1)

        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            #torch.nn.init.kaiming_normal(param)
            param.data = fanin_init(param.data.size())
        self.cuda()

    def actor_forward(self, state):
        #Primitive sub-network
        out = F.tanh(self.w_actor1.mm(state))
        out = F.tanh(self.w_actor2.mm(out))
        #out = F.leaky_relu(self.w_actor3.mm(out))
        #out = F.leaky_relu(self.w_actor4.mm(out))
        #out = F.relu(self.w_actor5.mm(out))
        out = F.sigmoid(self.w_actor6.mm(out))


        # action_softmax = F.log_softmax(prim2[0:-1,:], dim=0)
        # comm = torch.log(prim2[-1:,:])
        # out = torch.cat((action_softmax, comm))



        #Comm module
        #comm1 = F.tanh(self.w_actor_comm1.mm(state))
        #out = F.sigmoid(self.w_actor_comm2.mm(out))
        #out = F.log_softmax(out, dim=0)

        #LOG SOFTMAX
        out = F.log_softmax(out, dim=0)
        return out

    def critic_forward(self, state):
        out = F.tanh(self.w_critic1.mm(state))
        out = F.tanh(self.w_critic2.mm(out))
        #out = F.leaky_relu(self.w_critic3.mm(out))
        #out = F.leaky_relu(self.w_critic4.mm(out))
        #out = F.relu(self.w_critic5.mm(out))
        out = F.sigmoid(self.w_critic6.mm(out))
        return out

    def reset(self, batch_size=1):
        #self.mmu.reset(batch_size)
        pass

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

        #Rover path trace (viz)
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.params.action_dim)] for _ in range(self.params.num_rover)]

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
            for i in range(self.params.num_poi):
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
            for i in range(self.params.num_poi):
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
        self.rover_path = [[(loc[0], loc[1])] for loc in self.rover_pos]
        self.action_seq = [[0.0 for _ in range(self.params.action_dim)] for _ in range(self.params.num_rover)]

    def get_state(self, rover_id, ledger):
        if self.util_macro[rover_id][0]: #If currently active
            if self.util_macro[rover_id][1] == False: #Not first time activate (Not Is-activated-now)
                return np.zeros((720/self.params.angle_res + 5, 1)) -10000 #If macro return none
            else:
                self.util_macro[rover_id][1] = False  # Turn off is_activated_now?

        #return mod.unsqueeze(np.array([self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]]), 1)

        self_x = self.rover_pos[rover_id][0]; self_y = self.rover_pos[rover_id][1]

        rover_state = [0.0 for _ in range(360 / self.params.angle_res)]
        poi_state = [0.0 for _ in range(360 / self.params.angle_res)]
        temp_poi_dist_list = [[] for _ in range(360 / self.params.angle_res)]
        temp_rover_dist_list = [[] for _ in range(360 / self.params.angle_res)]

        # Log all distance into brackets for POIs
        x2 = -1.0; y2 = 0.0
        for loc, status in zip(self.poi_pos, self.poi_status):
            if status == True: continue #If accessed ignore

            x1 = loc[0] - self_x; y1 = loc[1] - self_y
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            if dist > self.params.obs_radius: continue #Observability radius

            bracket = int(angle / self.params.angle_res)
            temp_poi_dist_list[bracket].append(dist)

        # Log all distance into brackets for other drones
        for id, loc, in enumerate(self.rover_pos):
            if id == rover_id: continue #Ignore self

            x1 = loc[0] - self_x; y1 = loc[1] - self_y
            angle, dist = self.get_angle_dist(x1, y1, x2, y2)
            if dist > self.params.obs_radius: continue #Observability radius

            bracket = int(angle / self.params.angle_res)
            temp_rover_dist_list[bracket].append(dist)


        ####Encode the information onto the state
        for bracket in range(int(360 / self.params.angle_res)):
            # POIs
            num_poi = len(temp_poi_dist_list[bracket])
            if num_poi > 0:
                if self.params.sensor_model == 1: poi_state[bracket] = sum(temp_poi_dist_list[bracket]) / num_poi #Density Sensor
                else: poi_state[bracket] = min(temp_poi_dist_list[bracket])  #Minimum Sensor
            else: poi_state[bracket] = -1.0

            #Rovers
            num_rover = len(temp_rover_dist_list[bracket])
            if num_rover > 0:
                if self.params.sensor_model == 1: rover_state[bracket] = sum(temp_rover_dist_list[bracket]) / num_rover #Density Sensor
                else: rover_state[bracket] = min(temp_rover_dist_list[bracket]) #Minimum Sensor
            else: rover_state[bracket] = -1.0

        state = rover_state + poi_state #Append rover and poi to form the full state

        #Append wall info
        state = state + [-1.0, -1.0, -1.0, -1.0]
        if self_x <= self.params.obs_radius: state[-4] = self_x
        if self.params.dim_x - self_x <= self.params.obs_radius: state[-3] = self.params.dim_x - self_x
        if self_y <= self.params.obs_radius :state[-2] = self_y
        if self.params.dim_y - self_y <= self.params.obs_radius: state[-1] = self.params.dim_y - self_y

        #Add ledger and state vars
        closest_loc, min_dist = ledger.get_knn([self_x, self_y])
        if closest_loc[0] == None: closest_loc[0], closest_loc[1] = self_x, self_y
        self.ledger_closest[rover_id] = [closest_loc[0], closest_loc[1]]

        #if rover_id == 1: state =  np.zeros((720 / self.params.angle_res)) - 2 #TODO TEST
        state = state + [min_dist]
        state = unsqueeze(np.array(state), 1)
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

        for rover_id in range(self.params.num_rover):
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

        #Append rover path
        for rover_id in range(self.params.num_rover):
            self.rover_path[rover_id].append((self.rover_pos[rover_id][0], self.rover_pos[rover_id][1]))

        #Track rover actions
        for rover_id, action in enumerate(joint_action):
            self.action_seq[rover_id][action] += 1.0

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

    def trace_viz(self):
        # Visualize
        grid = [['-' for _ in range(self.dim_x)] for _ in range(self.dim_y)]

        drone_symbol_bank = ["0", "1", '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
        # Draw in rover path
        for rover_id in range(self.params.num_rover):
            for time in range(self.params.num_timestep):
                x = int(self.rover_path[rover_id][time][0]);
                y = int(self.rover_path[rover_id][time][1])
                # print x,y
                grid[x][y] = drone_symbol_bank[rover_id]

        # Draw in food
        for loc, status in zip(self.poi_pos, self.poi_status):
            x = int(loc[0]);
            y = int(loc[1])
            marker = 'I' if status else 'A'
            grid[x][y] = marker

        for row in grid:
            print row
        print

        print 'Action Diversity', self.action_seq

        # print agent.ledger.ledger
        print '------------------------------------------------------------------------'


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    #v = 1. / np.sqrt(fanin)
    v = 0.008
    return torch.Tensor(size).uniform_(-v, v)

def to_numpy(var):
    return var.cpu().data.numpy()

def to_tensor(ndarray, volatile=False, requires_grad=False):
    return Variable(torch.from_numpy(ndarray).float(), volatile=volatile, requires_grad=requires_grad).cuda()

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

def add_experience(state, new_state, action, reward, agent):
    agent.replay_buffer.add(1.0, [state, new_state, action, reward])





class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class GD_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(GD_MMU, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.memory_size = memory_size;
        self.output_size = output_size

        # Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( hidden_size, output_size), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size ), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size), requires_grad=1)

        # Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size ), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Memory Decoder
        self.w_decoder = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size ), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Memory Encoder
        self.w_encoder = Parameter(torch.rand(memory_size, hidden_size), requires_grad=1)


        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()


    def graph_compute(self, input, rec_output, memory, batch_size):

        # Input process
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))  # Block Input
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(memory) ) #Input gate + self.w_rec_inpgate.mm(rec_output)


        # Read from memory
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_mem_readgate.mm(memory) + self.w_rec_readgate.mm(rec_output))
        decoded_mem = self.w_decoder.mm(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp  * inp_gate

        # Update memory
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(memory) + self.w_rec_writegate.mm(rec_output))  # #Write gate
        encoded_update = F.tanh(self.w_encoder.mm(hidden_act))
        memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update


        return hidden_act, memory

    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True































#Extra
class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, num_hid = 50, num_mem=50, init_w=3e-3):
        super(Actor, self).__init__()

        self.fc1 = Parameter(torch.rand(num_hid, nb_states), requires_grad=1)
        self.mmu = GD_MMU(num_hid, num_hid, num_mem, num_hid)
        self.fc2 = Parameter(torch.rand(nb_actions, num_hid), requires_grad=1)
        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, state, action):
        out = self.fc1.mm(state)
        out = self.mmu.forward(out)
        out = self.fc2.mm(out)
        return out

    def reset(self, batch_size=1):
        self.mmu.reset(batch_size)

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, num_hid = 50, num_mem=50):
        super(Critic, self).__init__()

        self.fc1 = Parameter(torch.rand(num_hid, nb_states), requires_grad=1)
        self.mmu = GD_MMU(num_hid+nb_actions, num_hid, num_mem, num_hid)
        self.fc2 = Parameter(torch.rand(1, num_hid), requires_grad=1)
        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, state, action):
        out = self.fc1.mm(state)
        out = torch.cat([out,action],1)
        out = self.mmu.forward(out)
        out = self.fc2.mm(out)
        return out

    def reset(self, batch_size=1):
        self.mmu.reset(batch_size)

class bckGD_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(GD_MMU, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.memory_size = memory_size;
        self.output_size = output_size

        # Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size+1), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( hidden_size, output_size+1), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size + 1), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size+1), requires_grad=1)

        # Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Memory Decoder
        self.w_decoder = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Memory Encoder
        self.w_encoder = Parameter(torch.rand(memory_size, hidden_size), requires_grad=1)

        # Memory init
        self.w_mem_init = Parameter(torch.rand(memory_size, 1), requires_grad=1)

        # Adaptive components
        self.mem = (self.w_mem_init.mm(Variable(torch.zeros(1, 1), requires_grad=1))).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        # Adaptive components
        self.mem = self.w_mem_init.mm(Variable(torch.zeros(1, batch_size), requires_grad=1).cuda())
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def prep_bias(self, mat, batch_size):
        return Variable(torch.cat((mat.cpu().data, torch.ones(self.ou, batch_size))).cuda())

    def graph_compute(self, input, rec_output, memory, batch_size):

        # Reshape add 1 for bias
        input = self.prep_bias(input, batch_size)
        rec_output = self.prep_bias(rec_output, batch_size)

        # Input process
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))  # Block Input
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(memory) ) #Input gate + self.w_rec_inpgate.mm(rec_output)


        # Read from memory
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_mem_readgate.mm(memory) + self.w_rec_readgate.mm(rec_output))
        decoded_mem = self.w_decoder.mm(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp  * inp_gate

        # Update memory
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(memory) + self.w_rec_writegate.mm(rec_output))  # #Write gate
        encoded_update = F.tanh(self.w_encoder.mm(hidden_act))
        memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update


        return hidden_act, memory

    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True





class Stacked_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Stacked_MMU, self).__init__()

        # Define model
        # self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embeddings = nn.Embedding(n_vocab + 1, embedding_dim)
        self.mmu1 = mod.GD_MMU(embedding_dim, hidden_size, memory_size, hidden_size)
        self.mmu2 = mod.GD_MMU(hidden_size, hidden_size, memory_size, hidden_size)
        # self.mmu3 = mod.GD_MMU(hidden_size, hidden_size, memory_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        # self.dropout3 = nn.Dropout(0.1)


        # self.w_out1 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        # self.w_out2 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out3 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, input):
        embeds = self.embeddings(input)
        mmu1_out = self.mmu1.forward(torch.t(embeds))
        mmu1_out = self.dropout1(mmu1_out)
        mmu2_out = self.mmu2.forward(mmu1_out)
        mmu2_out = self.dropout2(mmu2_out)
        # mmu3_out = self.mmu3.forward(mmu2_out)
        # mmu3_out = self.dropout3(mmu3_out)

        out = self.w_out3.mm(mmu2_out)  # + self.w_out2.mm(mmu2_out) + self.w_out1.mm(mmu1_out)
        out = F.log_softmax(torch.t(out))
        return out

    def reset(self, batch_size):
        # self.poly.reset(batch_size)
        self.mmu1.reset(batch_size)
        self.mmu2.reset(batch_size)
        # self.mmu3.reset(batch_size)

class Single_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(Single_MMU, self).__init__()

        # Define model
        self.embeddings = nn.Embedding(n_vocab + 1, embedding_dim)
        self.mmu = mod.GD_MMU(embedding_dim, hidden_size, memory_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.w_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, input):
        embeds = self.embeddings(input)
        mmu_out = self.mmu.forward(torch.t(embeds))
        mmu_out = self.dropout(mmu_out)
        out = self.w_out.mm(mmu_out)
        out = F.log_softmax(torch.t(out))
        return out

    def reset(self, batch_size):
        # self.poly.reset(batch_size)
        self.mmu.reset(batch_size)

class Stacked_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Stacked_LSTM, self).__init__()

        # Define model
        # self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embeddings = nn.Embedding(n_vocab + 1, embedding_dim)
        self.lstm1 = mod.GD_LSTM(embedding_dim, hidden_size, memory_size, hidden_size)
        self.lstm2 = mod.GD_LSTM(hidden_size, hidden_size, memory_size, hidden_size)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        # self.bnorm1 = nn.BatchNorm2d(hidden_size)
        # self.w_out1 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)
        self.w_out2 = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, input):
        embeds = self.embeddings(input)
        lstm1_out = self.lstm1.forward(torch.t(embeds))
        lstm1_out = self.dropout1(lstm1_out)
        # lstm1_out = self.bnorm1(lstm1_out)
        lstm2_out = self.lstm2.forward(lstm1_out)
        lstm2_out = self.dropout1(lstm2_out)

        out = self.w_out2.mm(lstm2_out)  # + self.w_out2.mm(lstm2_out) + self.w_out1.mm(lstm1_out)
        out = F.log_softmax(torch.t(out))
        return out

    def reset(self, batch_size):
        # self.poly.reset(batch_size)
        self.lstm1.reset(batch_size)
        self.lstm2.reset(batch_size)

class Single_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size, n_vocab):
        super(Single_LSTM, self).__init__()

        # Define model
        # self.poly = mod.GD_polynet(input_size, hidden_size, hidden_size, hidden_size, None)
        self.embeddings = nn.Embedding(n_vocab + 1, embedding_dim)
        self.lstm = mod.GD_LSTM(embedding_dim, hidden_size, memory_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.w_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def forward(self, input):
        embeds = self.embeddings(input)
        lstm_out = self.lstm.forward(torch.t(embeds))
        lstm_out = self.dropout(lstm_out)

        out = self.w_out.mm(lstm_out)
        out = F.log_softmax(torch.t(out))
        return out

    def reset(self, batch_size):
        # self.poly.reset(batch_size)
        self.lstm.reset(batch_size)














#Polynet bundle
class Polynet(nn.Module):
    def __init__(self, input_size, h1, h2, output_size, output_activation):
        super(Polynet, self).__init__()

        self.input_size = input_size; self.h1 = h1; self.h2 = h2; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = expit
        elif output_activation == 'tanh': self.output_activation = np.tanh
        else: self.output_activation = None

        #Weights
        self.w1 = np.mat(np.random.normal(0,1, (h1, input_size)))
        self.w_poly = np.mat(np.ones((h2, h1)))
        self.w2 = np.mat(np.random.normal(0, 1, (output_size, h2)))

        self.param_dict = {'w1': self.w1,
                           'w_poly': self.w_poly,
                           'w2': self.w2}
        self.gd_net = GD_polynet(input_size, h1, h2, output_size, output_activation)


    def forward(self, input):
        batch_size = input.shape[1]

        first_out = np.dot(self.w1, input) #Linear transform
        first_out = np.multiply(first_out, (first_out > 0.01)) #First dense layer with thresholding activation (Relu except 0.01 translated to 0.01)

        #Polynomial Operation
        poly_out = self.poly_op(first_out, batch_size) #Polynomial dot product
        output = np.dot(self.w2, poly_out) #Output dense layer
        if self.output_activation != None: output = self.output_activation(output)
        return output

    def poly_op(self, inp, batch_size):
        poly_out = np.mat(np.zeros((self.h2, batch_size)))
        for i, node in enumerate(self.w_poly):
            batch_poly = self.batch_copy(node, batch_size, axis=0)
            node_act = np.sum(np.power(inp, np.transpose(batch_poly)), axis=0)
            poly_out[i,:] = node_act

        return poly_out

    def batch_copy(self, mat, batch_size, axis):
        padded_mat = np.copy(mat)
        for _ in range(batch_size - 1): padded_mat = np.concatenate((padded_mat, mat), axis=axis)
        return padded_mat

    def from_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            params[key][:] = gd_params[key].cpu().numpy()

    def to_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            gd_params[key][:] = params[key]

    def reset(self, batch_size):
        return

class GD_polynet(nn.Module):
    def __init__(self, input_size, h1, h2, output_size, output_activation):
        super(GD_polynet, self).__init__()

        self.input_size = input_size; self.output_size = output_size
        if output_activation == 'sigmoid': self.output_activation = F.sigmoid
        elif output_activation == 'tanh': self.output_activation = F.tanh
        else: self.output_activation = None

        #Weights
        self.w1 = Parameter(torch.rand(h1, input_size), requires_grad=1)
        self.w_poly = Parameter(torch.rand(h2, h1), requires_grad=1)
        self.w2 = Parameter(torch.rand(output_size, h2), requires_grad=1)

        #Initialize weights except for poly weights which are initialized to all 1s
        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            #torch.nn.init.orthogonal(param)
            #torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)
        #self.w_poly = Parameter(torch.ones(h2, h1), requires_grad=1) +
        self.w_poly.data += 1.0


    def forward(self, input):
        first_out = F.threshold(self.w1.mm(input), 0.01, 0.01) #First dense layer with thresholding activation (Relu except 0 translated to 0.1)

        #Polynomial operation
        poly1 = torch.t(first_out).pow(self.w_poly)
        poly_out = torch.sum(poly1, 1).unsqueeze(1)

        #Output dense layer
        output = self.w2.mm(poly_out)
        if self.output_activation != None: output = self.output_activation(output)
        return output

    #TODO Batch Process for GD_Polynet
    def reset(self, batch_size):
        return


#MMU Bundle
class MMU:
    def __init__(self, num_input, num_hnodes, num_memory, num_output, output_activation, mean = 0, std = 1):
        self.num_input = num_input; self.num_output = num_output; self.num_hnodes = num_hnodes; self.num_mem = num_memory
        self.output_activation = output_activation

        #Input gate
        self.w_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_input)))
        self.w_rec_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_output)))
        self.w_mem_inpgate = np.mat(np.random.normal(mean, std, (num_hnodes, num_memory)))

        #Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, (num_hnodes, num_input)))
        self.w_rec_inp = np.mat(np.random.normal(mean, std, (num_hnodes, num_output)))

        #Read gate
        self.w_readgate = np.mat(np.random.normal(mean, std, (num_memory, num_input)))
        self.w_rec_readgate = np.mat(np.random.normal(mean, std, (num_memory, num_output)))
        self.w_mem_readgate = np.mat(np.random.normal(mean, std, (num_memory, num_memory)))

        #Memory Decoder
        self.w_decoder = np.mat(np.random.normal(mean, std, (num_hnodes, num_memory)))

        #Memory write gate
        self.w_writegate = np.mat(np.random.normal(mean, std, (num_memory, num_input)))
        self.w_rec_writegate = np.mat(np.random.normal(mean, std, (num_memory, num_output)))
        self.w_mem_writegate = np.mat(np.random.normal(mean, std, (num_memory, num_memory)))

        # Memory Encoder
        self.w_encoder= np.mat(np.random.normal(mean, std, (num_memory, num_hnodes)))

        #Output weights
        self.w_hid_out= np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))

        #Biases
        self.w_input_gate_bias = np.mat(np.zeros((num_hnodes, 1)))
        self.w_block_input_bias = np.mat(np.zeros((num_hnodes, 1)))
        self.w_readgate_bias = np.mat(np.zeros((num_memory, 1)))
        self.w_writegate_bias = np.mat(np.zeros((num_memory, 1)))

        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((num_output, 1)))
        self.memory = np.mat(np.zeros((num_memory, 1)))

        self.param_dict = {'w_inpgate': self.w_inpgate,
                           'w_rec_inpgate': self.w_rec_inpgate,
                           'w_mem_inpgate': self.w_mem_inpgate,
                           'w_inp': self.w_inp,
                           'w_rec_inp': self.w_rec_inp,
                            'w_readgate': self.w_readgate,
                            'w_rec_readgate': self.w_rec_readgate,
                            'w_mem_readgate': self.w_mem_readgate,
                           'w_decoder' : self.w_decoder,
                            'w_writegate': self.w_writegate,
                            'w_rec_writegate': self.w_rec_writegate,
                            'w_mem_writegate': self.w_mem_writegate,
                           'w_encoder': self.w_encoder,
                           'w_hid_out': self.w_hid_out,
                            'w_input_gate_bias': self.w_input_gate_bias,
                           'w_block_input_bias': self.w_block_input_bias,
                            'w_readgate_bias': self.w_readgate_bias,
                           'w_writegate_bias': self.w_writegate_bias}

        self.gd_net = GD_MMU(num_input, num_hnodes, num_memory, num_output, output_activation) #Gradient Descent Net

    def forward(self, input): #Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)
        #Input gate
        input_gate_out = expit(np.dot(self.w_inpgate, input)+ np.dot(self.w_rec_inpgate, self.output) + np.dot(self.w_mem_inpgate, self.memory) + self.w_input_gate_bias)

        #Input processing
        block_input_out = expit(np.dot(self.w_inp, input) + np.dot(self.w_rec_inp, self.output) + self.w_block_input_bias)

        #Gate the Block Input and compute the final input out
        input_out = np.multiply(input_gate_out, block_input_out)

        #Read Gate
        read_gate_out = expit(np.dot(self.w_readgate, input) + np.dot(self.w_rec_readgate, self.output) + np.dot(self.w_mem_readgate, self.memory) + self.w_readgate_bias)

        #Compute hidden activation - processing hidden output for this iteration of net run
        decoded_mem = np.dot(self.w_decoder, np.multiply(read_gate_out, self.memory))
        hidden_act =  decoded_mem + input_out

        #Write gate (memory cell)
        write_gate_out = expit(np.dot(self.w_writegate, input)+ np.dot(self.w_rec_writegate, self.output) + np.dot(self.w_mem_writegate, self.memory) + self.w_writegate_bias)

        #Write to memory Cell - Update memory
        encoded_update = np.dot(self.w_encoder, hidden_act)
        self.memory += np.multiply(write_gate_out, encoded_update)

        #Compute final output
        self.output = np.dot(self.w_hid_out, hidden_act)
        if self.output_activation == 'tanh': self.output = np.tanh(self.output)
        if self.output_activation == 'sigmoid': self.output = expit(self.output)
        return self.output

    def reset(self, batch_size):
        #Adaptive components (plastic with network running)
        self.output = np.mat(np.zeros((self.num_output, batch_size)))
        self.memory = np.mat(np.zeros((self.num_mem, batch_size)))

    def from_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            params[key][:] = gd_params[key].cpu().numpy()

    def to_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            gd_params[key][:] = params[key]

class bGD_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(GD_MMU, self).__init__()

        self.input_size = input_size; self.hidden_size = hidden_size; self.memory_size = memory_size; self.output_size = output_size

        #Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size+1), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( hidden_size, output_size+1), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        #Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size+1), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size+1), requires_grad=1)

        #Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size+1), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Memory Decoder
        self.w_decoder = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        #Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size+1), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        #Memory Encoder
        self.w_encoder = Parameter(torch.rand(memory_size, hidden_size), requires_grad=1)

        #Memory init
        self.w_mem_init = Parameter(torch.rand(memory_size, 1), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.ones(1, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            #torch.nn.init.xavier_normal(param)
            #torch.nn.init.orthogonal(param)
            #torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

        #Gates to 1
        # self.w_writegate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
        # self.w_rec_writegate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
        # self.w_mem_writegate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
        # self.w_readgate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
        # self.w_rec_readgate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
        # self.w_mem_readgate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
        # self.w_inpgate = Parameter(torch.ones(hidden_size, input_size), requires_grad=1)
        # self.w_rec_inpgate = Parameter(torch.ones(hidden_size, output_size), requires_grad=1)
        # self.w_mem_inpgate = Parameter(torch.ones(hidden_size, memory_size), requires_grad=1)


    def reset(self, batch_size):
        # Adaptive components
        self.mem = self.w_mem_init.mm(Variable(torch.zeros(1, batch_size), requires_grad=1).cuda())
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def prep_bias(self, mat, batch_size):
        return Variable(torch.cat((mat.cpu().data, torch.ones(1, batch_size))).cuda())

    def bgraph_compute(self, input, rec_output, memory, batch_size):
        #Reshape add 1 for bias
        input = self.prep_bias(input, batch_size); rec_output = self.prep_bias(rec_output, batch_size); mem = self.prep_bias(memory, batch_size)

        #Input process
        block_inp = F.tanh(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output)) #Block Input
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(rec_output)) #Input gate


        #Read from memory
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(mem))
        decoded_mem = self.w_decoder.mm(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = block_inp * inp_gate + decoded_mem

        #Update memory
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(rec_output)) # #Write gate
        encoded_update = F.tanh(self.w_encoder.mm(hidden_act))
        memory = memory + write_gate_out * encoded_update

        return hidden_act, memory

    def graph_compute(self, input, rec_output, memory, batch_size):

        #Reshape add 1 for bias
        input = self.prep_bias(input, batch_size); rec_output = self.prep_bias(rec_output, batch_size)

        #Input process
        block_inp = F.tanh(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output)) #Block Input
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(memory) + self.w_rec_inpgate.mm(rec_output)) #Input gate


        #Read from memory
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(memory))
        decoded_mem = self.w_decoder.mm(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = block_inp * inp_gate + decoded_mem

        #Update memory
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(memory) + self.w_rec_writegate.mm(rec_output)) # #Write gate
        encoded_update = F.tanh(self.w_encoder.mm(hidden_act))
        #memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update
        memory = memory + encoded_update

        return hidden_act, memory


    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

class oldGD_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(GD_LSTM, self).__init__()

        self.input_size = input_size; self.hidden_size = hidden_size; self.memory_size = memory_size; self.output_size = output_size

        # Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size+1), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(hidden_size, output_size+1), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size+1), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size+1), requires_grad=1)

        # Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size+1), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size+1), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)



        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

            # Gates to 1
            # self.w_writegate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
            # self.w_rec_writegate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
            # self.w_mem_writegate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
            # self.w_readgate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
            # self.w_rec_readgate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
            # self.w_mem_readgate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
            # self.w_inpgate = Parameter(torch.ones(hidden_size, input_size), requires_grad=1)
            # self.w_rec_inpgate = Parameter(torch.ones(hidden_size, output_size), requires_grad=1)
            # self.w_mem_inpgate = Parameter(torch.ones(hidden_size, memory_size), requires_grad=1)

    def prep_bias(self, mat, batch_size):
        return Variable(torch.cat((mat.cpu().data, torch.ones(1, batch_size))).cuda())

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def graph_compute(self, input, rec_output, mem, batch_size):
        #Reshape add 1 for bias
        input = self.prep_bias(input, batch_size); rec_output = self.prep_bias(rec_output, batch_size)

        # Block Input
        block_inp = F.tanh(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))  # + self.w_block_input_bias)

        # Input gate
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(rec_output))  # + self.w_input_gate_bias)

        # Input out
        inp_out = block_inp * inp_gate

        # Read gate
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(mem))  # + self.w_readgate_bias) * mem

        # Output gate
        out_gate = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(rec_output))  # + self.w_writegate_bias)

        # Compute new mem
        mem = inp_out + read_gate_out * mem
        out = out_gate * mem


        return out, mem


    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

class bckGD_MMU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(GD_MMU, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.memory_size = memory_size;
        self.output_size = output_size

        # Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size+1), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand( hidden_size, output_size+1), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size + 1), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size+1), requires_grad=1)

        # Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Memory Decoder
        self.w_decoder = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size+1), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Memory Encoder
        self.w_encoder = Parameter(torch.rand(memory_size, hidden_size), requires_grad=1)

        # Memory init
        self.w_mem_init = Parameter(torch.rand(memory_size, 1), requires_grad=1)

        # Adaptive components
        self.mem = self.w_mem_init.mm(Variable(torch.zeros(1, 1), requires_grad=1))
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        # Adaptive components
        self.mem = self.w_mem_init.mm(Variable(torch.zeros(1, batch_size), requires_grad=1).cuda())
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def prep_bias(self, mat, batch_size):
        return Variable(torch.cat((mat.cpu().data, torch.ones(1, batch_size))).cuda())

    def graph_compute(self, input, rec_output, memory, batch_size):

        # Reshape add 1 for bias
        input = self.prep_bias(input, batch_size)
        rec_output = self.prep_bias(rec_output, batch_size)

        # Input process
        block_inp = F.sigmoid(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))  # Block Input
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(memory) ) #Input gate + self.w_rec_inpgate.mm(rec_output)


        # Read from memory
        read_gate_out = F.sigmoid(self.w_readgate.mm(input) + self.w_mem_readgate.mm(memory) + self.w_rec_readgate.mm(rec_output))
        decoded_mem = self.w_decoder.mm(read_gate_out * memory)

        # Compute hidden activation
        hidden_act = decoded_mem + block_inp  * inp_gate

        # Update memory
        write_gate_out = F.sigmoid(self.w_writegate.mm(input) + self.w_mem_writegate.mm(memory) + self.w_rec_writegate.mm(rec_output))  # #Write gate
        encoded_update = F.tanh(self.w_encoder.mm(hidden_act))
        memory = (1 - write_gate_out) * memory + write_gate_out * encoded_update


        return hidden_act, memory

    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True

class GD_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(GD_LSTM, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.memory_size = memory_size;
        self.output_size = output_size

        # Input gate
        self.w_inpgate = Parameter(torch.rand(hidden_size, input_size + 1), requires_grad=1)
        self.w_rec_inpgate = Parameter(torch.rand(hidden_size, output_size + 1), requires_grad=1)
        self.w_mem_inpgate = Parameter(torch.rand(hidden_size, memory_size), requires_grad=1)

        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size + 1), requires_grad=1)
        self.w_rec_inp = Parameter(torch.rand(hidden_size, output_size + 1), requires_grad=1)

        # Read Gate
        self.w_readgate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_readgate = Parameter(torch.rand(memory_size, output_size + 1), requires_grad=1)
        self.w_mem_readgate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Write Gate
        self.w_writegate = Parameter(torch.rand(memory_size, input_size + 1), requires_grad=1)
        self.w_rec_writegate = Parameter(torch.rand(memory_size, output_size + 1), requires_grad=1)
        self.w_mem_writegate = Parameter(torch.rand(memory_size, memory_size), requires_grad=1)

        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, 1), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, 1), requires_grad=1).cuda()

        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

            # Gates to 1
            # self.w_writegate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
            # self.w_rec_writegate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
            # self.w_mem_writegate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
            # self.w_readgate = Parameter(torch.ones(memory_size, input_size), requires_grad=1)
            # self.w_rec_readgate = Parameter(torch.ones(memory_size, output_size), requires_grad=1)
            # self.w_mem_readgate = Parameter(torch.ones(memory_size, memory_size), requires_grad=1)
            # self.w_inpgate = Parameter(torch.ones(hidden_size, input_size), requires_grad=1)
            # self.w_rec_inpgate = Parameter(torch.ones(hidden_size, output_size), requires_grad=1)
            # self.w_mem_inpgate = Parameter(torch.ones(hidden_size, memory_size), requires_grad=1)

    def prep_bias(self, mat, batch_size):
        return Variable(torch.cat((mat.cpu().data, torch.ones(1, batch_size))).cuda())

    def reset(self, batch_size):
        # Adaptive components
        self.mem = Variable(torch.zeros(self.memory_size, batch_size), requires_grad=1).cuda()
        self.out = Variable(torch.zeros(self.output_size, batch_size), requires_grad=1).cuda()

    def graph_compute(self, input, rec_output, mem, batch_size):
        # Reshape add 1 for bias
        input = self.prep_bias(input, batch_size);
        rec_output = self.prep_bias(rec_output, batch_size)

        # Block Input
        block_inp = F.tanh(self.w_inp.mm(input) + self.w_rec_inp.mm(rec_output))  # + self.w_block_input_bias)

        # Input gate
        inp_gate = F.sigmoid(self.w_inpgate.mm(input) + self.w_mem_inpgate.mm(mem) + self.w_rec_inpgate.mm(
            rec_output))  # + self.w_input_gate_bias)

        # Input out
        inp_out = block_inp * inp_gate

        # Read gate
        read_gate_out = F.sigmoid(
            self.w_readgate.mm(input) + self.w_rec_readgate.mm(rec_output) + self.w_mem_readgate.mm(
                mem))  # + self.w_readgate_bias) * mem

        # Output gate
        out_gate = F.sigmoid(
            self.w_writegate.mm(input) + self.w_mem_writegate.mm(mem) + self.w_rec_writegate.mm(
                rec_output))  # + self.w_writegate_bias)

        # Compute new mem
        mem = inp_out + read_gate_out * mem
        out = out_gate * mem

        return out, mem

    def forward(self, input):
        batch_size = input.data.shape[-1]
        self.out, self.mem = self.graph_compute(input, self.out, self.mem, batch_size)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


#FF Bundle
class FF:
    def __init__(self, num_input, num_hnodes, num_output, output_activation, mean=0, std=1):
        self.num_input = num_input;
        self.num_output = num_output;
        self.num_hnodes = num_hnodes;
        self.output_activation = output_activation

        # Block Input
        self.w_inp = np.mat(np.random.normal(mean, std, (num_hnodes, num_input)))

        # Output weights
        self.w_hid_out = np.mat(np.random.normal(mean, std, (num_output, num_hnodes)))

        # Biases
        self.w_inp_bias = np.mat(np.zeros((num_hnodes, 1)))
        self.w_output_bias = np.mat(np.zeros((num_output, 1)))



        self.param_dict = {'w_inp': self.w_inp,
                           'w_hid_out': self.w_hid_out,
                           'w_inp_bias': self.w_inp_bias,
                           'w_output_bias': self.w_output_bias}

        self.gd_net = GD_FF(num_input, num_hnodes, num_output, output_activation)  # Gradient Descent Net

    def forward(self, input):  # Feedforwards the input and computes the forward pass of the network
        input = np.mat(input)

        # Hidden activations
        hidden_act = expit(np.dot(self.w_inp, input) + self.w_inp_bias)

        # Compute final output
        self.output = np.dot(self.w_hid_out, hidden_act) + self.w_output_bias
        if self.output_activation == 'tanh': self.output = np.tanh(self.output)
        if self.output_activation == 'sigmoid': self.output = expit(self.output)
        return self.output

    def reset(self, batch_size):
        return

    def from_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            params[key][:] = gd_params[key].cpu().numpy()

    def to_gdnet(self):
        self.gd_net.reset(batch_size=1)
        self.reset(batch_size=1)

        gd_params = self.gd_net.state_dict()  # GD-Net params
        params = self.param_dict  # Self params

        keys = self.gd_net.state_dict().keys()  # Common keys
        for key in keys:
            gd_params[key][:] = params[key]

class GD_FF(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_activation):
        super(GD_FF, self).__init__()

        self.input_size = input_size;
        self.hidden_size = hidden_size;
        self.output_size = output_size
        if output_activation == 'sigmoid':
            self.output_activation = F.sigmoid
        elif output_activation == 'tanh':
            self.output_activation = F.tanh
        else:
            self.output_activation = None


        # Block Input
        self.w_inp = Parameter(torch.rand(hidden_size, input_size), requires_grad=1)

        # Output weights
        self.w_hid_out = Parameter(torch.rand(output_size, hidden_size), requires_grad=1)


        for param in self.parameters():
            # torch.nn.init.xavier_normal(param)
            # torch.nn.init.orthogonal(param)
            # torch.nn.init.sparse(param, sparsity=0.5)
            torch.nn.init.kaiming_normal(param)

    def reset(self, batch_size):
        return

    def graph_compute(self, input):
        # Compute hidden activations
        hidden_act = F.sigmoid(self.w_inp.mm(input))  # + self.w_block_input_bias)

        #Compute Output
        output = self.w_hid_out.mm(hidden_act)
        if self.output_activation != None: output = self.output_activation(output)

        return output

    def forward(self, input):
        self.out = self.graph_compute(input)
        return self.out

    def turn_grad_on(self):
        for param in self.parameters():
            param.requires_grad = True
            param.volatile = False

    def turn_grad_off(self):
        for param in self.parameters():
            param.requires_grad = False
            param.volatile = True


#Neuroevolution SSNE
class SSNE:
    def __init__(self, parameters):
        self.current_gen = 0
        self.parameters = parameters;
        self.population_size = self.parameters.pop_size;
        self.num_elitists = int(self.parameters.elite_fraction * parameters.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2):
        keys = list(gene1.param_dict.keys())

        # References to the variable tensors
        W1 = gene1.param_dict
        W2 = gene2.param_dict
        num_variables = len(W1)
        if num_variables != len(W2): print 'Warning: Genes for crossover might be incompatible'

        # Crossover opertation [Indexed by column, not rows]
        num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
        for i in range(num_cross_overs):
            tensor_choice = fastrand.pcg32bounded(num_variables)  # Choose which tensor to perturb
            receiver_choice = random.random()  # Choose which gene to receive the perturbation
            if receiver_choice < 0.5:
                ind_cr = fastrand.pcg32bounded(W1[keys[tensor_choice]].shape[-1])  #
                W1[keys[tensor_choice]][:, ind_cr] = W2[keys[tensor_choice]][:, ind_cr]
                #W1[keys[tensor_choice]][ind_cr, :] = W2[keys[tensor_choice]][ind_cr, :]
            else:
                ind_cr = fastrand.pcg32bounded(W2[keys[tensor_choice]].shape[-1])  #
                W2[keys[tensor_choice]][:, ind_cr] = W1[keys[tensor_choice]][:, ind_cr]
                #W2[keys[tensor_choice]][ind_cr, :] = W1[keys[tensor_choice]][ind_cr, :]

    def mutate_inplace(self, gene):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        # References to the variable keys
        keys = list(gene.param_dict.keys())
        W = gene.param_dict
        num_structures = len(keys)
        ssne_probabilities = np.random.uniform(0,1,num_structures)*2

        for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure apart from poly
            if random.random()<ssne_prob:
                num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                for _ in range(num_mutations):
                    ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    random_num = random.random()

                    if random_num < super_mut_prob:  # Super Mutation probability
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[key][ind_dim1, ind_dim2])
                    elif random_num < reset_prob:  # Reset probability
                        W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)

                    else:  # mutauion even normal
                        W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][ind_dim1, ind_dim2])

                    # Regularization hard limit
                    W[key][ind_dim1, ind_dim2] = self.regularize_weight(W[key][ind_dim1, ind_dim2], self.parameters.weight_magnitude_limit)

    def trial_mutate_inplace(self, hive):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_prob = 0.05

        for drone in hive.all_drones:
            # References to the variable keys
            keys = list(drone.param_dict.keys())
            W = drone.param_dict
            num_structures = len(keys)
            ssne_probabilities = np.random.uniform(0,1,num_structures)*2

            for ssne_prob, key in zip(ssne_probabilities, keys): #For each structure
                if random.random()<ssne_prob:

                    mut_matrix = scipy_rand(W[key].shape[0], W[key].shape[1], density=num_mutation_frac, data_rvs=np.random.randn).A * mut_strength
                    W[key] += np.multiply(mut_matrix, W[key])


                    # num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * W[key].size)))  # Number of mutation instances
                    # for _ in range(num_mutations):
                    #     ind_dim1 = fastrand.pcg32bounded(W[key].shape[0])
                    #     ind_dim2 = fastrand.pcg32bounded(W[key].shape[-1])
                    #     random_num = random.random()
                    #
                    #     if random_num < super_mut_prob:  # Super Mutation probability
                    #         W[key][ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength *
                    #                                                                       W[key][
                    #                                                                           ind_dim1, ind_dim2])
                    #     elif random_num < reset_prob:  # Reset probability
                    #         W[key][ind_dim1, ind_dim2] = random.gauss(0, 1)
                    #
                    #     else:  # mutauion even normal
                    #         W[key][ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[key][
                    #                                                                           ind_dim1, ind_dim2])
                    #
                    #     # Regularization hard limit
                    #     W[key][ind_dim1, ind_dim2] = self.regularize_weight(
                    #         W[key][ind_dim1, ind_dim2])

    def copy_individual(self, master, replacee):  # Replace the replacee individual with master
            keys = master.param_dict.keys()
            for key in keys:
                replacee.param_dict[key][:] = master.param_dict[key]

    def reset_genome(self, gene):
        keys = gene.param_dict
        for key in keys:
            dim = gene.param_dict[key].shape
            gene.param_dict[key][:] = np.mat(np.random.uniform(-1, 1, (dim[0], dim[1])))

    def epoch(self, all_hives, fitness_evals):

        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        index_rank = self.list_argsort(fitness_evals); index_rank.reverse()
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard

        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        #Extinction step (Resets all the offsprings genes; preserves the elitists)
        if random.random() < self.parameters.extinction_prob: #An extinction event
            print
            print "######################Extinction Event Triggered#######################"
            print
            for i in offsprings:
                if random.random() < self.parameters.extinction_magnituide and not (i in elitist_index):  # Extinction probabilities
                    self.reset_genome(all_hives[i])

        # Figure out unselected candidates
        unselects = []; new_elitists = []
        for i in range(self.population_size):
            if i in offsprings or i in elitist_index:
                continue
            else:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            replacee = unselects.pop(0)
            new_elitists.append(replacee)
            self.copy_individual(master=all_hives[i], replacee=all_hives[replacee])

        # Crossover for unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists);
            off_j = random.choice(offsprings)
            self.copy_individual(master=all_hives[off_i], replacee=all_hives[i])
            self.copy_individual(master=all_hives[off_j], replacee=all_hives[j])
            self.crossover_inplace(all_hives[i], all_hives[j])

        # Crossover for selected offsprings
        for i, j in zip(offsprings[0::2], offsprings[1::2]):
            if random.random() < self.parameters.crossover_prob: self.crossover_inplace(all_hives[i], all_hives[j])

        # Mutate all genes in the population except the new elitists plus homozenize
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.parameters.mutation_prob: self.mutate_inplace(all_hives[i])


#Functions
def unpickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def pickle_object(obj, filename):
    with open(filename, 'wb') as output:
        cPickle.dump(obj, output, -1)

def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))



