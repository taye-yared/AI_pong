import numpy
import random
from MDP.MDP import MDP

class Simulator:
    
    def __init__(self, num_games=0, alpha_value=0, gamma_value=0, epsilon_value=0):
        '''
        Setup the Simulator with the provided values.
        :param num_games - number of games to be trained on.
        :param alpha_value - 1/alpha_value is the decay constant.
        :param gamma_value - Discount Factor.
        :param epsilon_value - Probability value for the epsilon-greedy approach.
        '''
        self.num_games = num_games       
        self.epsilon_value = epsilon_value       
        self.alpha_value = alpha_value       
        self.gamma_val = gamma_value
        self.q_table = {}
        self.mdp = MDP()
        self.hits = 0
        self.train_agent()


    def f_function(self):
        '''
        Choose action based on an epsilon greedy approach
        :return action selected
        '''
        rand_num = random.random()
        action_selected = 0
        if(rand_num < .04):
            action_selected = random.randint(0, 2)
        else:
            state = self.mdp.discretize_state()
            action_selected = numpy.argmax(numpy.array(self.get_table_val((state[0], state[1], state[2], state[3]))))
        
        return action_selected

    def train_agent(self):
        '''
        Train the agent over a certain number of games.
        '''
        for i in range(self.num_games):
            self.mdp.reset()
            self.play_game()
            if(i% 1000 == 0):
                print("Game ", i, " average ", int(self.hits/1000), " hits")
                self.hits = 0
    
    def play_game(self):
        '''
        Simulate an actual game till the agent loses.
        '''
        reward = 0
        while reward != -1:
            action = self.f_function()  # pick action
            prevState = self.mdp.discretize_state()
            reward = self.mdp.simulate_one_time_step(action)
            if reward != -1:
                if reward == 1:
                    self.hits += 1
                self.update_q(prevState, self.mdp.discretize_state(), reward, action)


    def update_q(self, prev_state, state, reward, action):
        if(self.q_table.get((prev_state[0], prev_state[1], prev_state[2], prev_state[3])) == None):
            self.q_table[(prev_state[0], prev_state[1], prev_state[2], prev_state[3])] = [0, 0, 0]
        prev_q_val = self.q_table[(prev_state[0], prev_state[1], prev_state[2], prev_state[3])][action]

        best_action = self.f_function()

        if(self.q_table.get((state[0], state[1], state[2], state[3])) == None):
            self.q_table[(state[0], state[1], state[2], state[3])] = [0, 0, 0]
        curr_q_val = self.q_table[(state[0], state[1], state[2], state[3])][best_action]

        prev_q_val = prev_q_val + self.alpha_value * (reward + self.gamma_val*curr_q_val - prev_q_val)
        self.q_table[(prev_state[0], prev_state[1], prev_state[2], prev_state[3])][action] = prev_q_val

    # Returns an array of size 3 for q-val of each move
    def get_table_val(self, index):
        if(self.q_table.get(index) != None):
            return self.q_table[index]
        else:
            self.q_table[index] = [0, 0, 0]
            return self.q_table[index]
