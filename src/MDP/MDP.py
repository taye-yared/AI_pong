import random
import math
import numpy

class MDP:

    def __init__(self, 
                 ball_x=None,
                 ball_y=None,
                 velocity_x=None,
                 velocity_y=None,
                 paddle_y=None):
        '''
        Setup MDP with the initial values provided.
        '''
        self.create_state(
            ball_x=ball_x,
            ball_y=ball_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            paddle_y=paddle_y
        )
        
        # the agent can choose between 3 actions - stay, up or down respectively.
        self.actions = [0, 0.04, -0.04]
        
    
    def create_state(self,
              ball_x=None,
              ball_y=None,
              velocity_x=None,
              velocity_y=None,
              paddle_y=None):
        '''
        Helper function for the initializer. Initialize member variables with provided or default values.
        '''
        self.paddle_height = 0.2
        self.ball_x = ball_x if ball_x != None else 0.5
        self.ball_y = ball_y if ball_y != None else 0.5
        self.velocity_x = velocity_x if velocity_x != None else 0.03
        self.velocity_y = velocity_y if velocity_y != None else 0.01
        self.paddle_y = 0.5
    
    def simulate_one_time_step(self, action_selected):
        '''
        :param action_selected - Current action to execute.
            0 - Do nothing
            1 - Move up
            2 - Move Down
        Perform the action on the current continuous state.
        '''
        if action_selected == 2:
            self.move_paddle_down()
        if action_selected == 1:
            self.move_paddle_up()

        self.increment_ball()
        reward = self.check_reward()
        # If ball movement causes a paddle bounce
        if reward == 1:
            self.ball_x = 2 - self.ball_x
            U = random.uniform(-.015, .015)
            V = random.uniform(-.03, .03)
            self.velocity_x = -self.velocity_x + U
            self.velocity_y += V
        if (self.velocity_x >= 0) & (self.velocity_x < .03):
            self.velocity_x  = .03
        if(self.velocity_x < 0) & (self.velocity_x > -.03):
            self.velocity_x  = -.03
        return reward




    
    def discretize_state(self):
        '''
        Convert the current continuous state to a discrete state.
        '''
        ret_val = numpy.zeros(4)
        discrete_ball_x = int(math.floor(self.ball_x*12))
        discrete_ball_y = int(math.floor(self.ball_y*12))
        discrete_ball_position = 12*discrete_ball_y+discrete_ball_x
        ret_val[0] = discrete_ball_position

        discrete_velocity_x = 0
        if self.velocity_x > 0:
            discrete_velocity_x = 1
        else:
            discrete_velocity_x = -1
        ret_val[1] = discrete_velocity_x

        discrete_velocity_y = 0
        if self.velocity_y > .015:
            discrete_velocity_y = 1
        if self.velocity_y < -.015:
            discrete_velocity_y = -1
        ret_val[2] = discrete_velocity_y

        discrete_paddle = int(math.floor(12 * self.paddle_y / (1 - self.paddle_height)))
        ret_val[3] = discrete_paddle-1
        if self.check_reward() == -1:
            return None
        return ret_val




    def move_paddle_down(self):
        if self.paddle_y + self.paddle_height > .96:
            self.paddle_y = 1 - self.paddle_height
        else:
            self.paddle_y += .04

    def move_paddle_up(self):
        if self.paddle_y < .04:
            self.paddle_y = 0
        else:
            self.paddle_y -= .04

    def check_reward(self):
        if self.ball_x >= 1:
            if(self.ball_y >= self.paddle_y) & (self.ball_y <= self.ball_y + self.paddle_height):
                return 1
            else:
                return -1
        else:
            return 0

    def increment_ball(self):
        self.ball_x += self.velocity_x

        if self.ball_x < 0:
            self.ball_x *= -1
            self.velocity_x *= -1

        self.ball_y += self.velocity_y

        if self.ball_y < 0:
            self.ball_y *= -1
            self.velocity_y *= -1

        if self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y *= -1



    def reset(self):
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.velocity_x = 0.03
        self.velocity_y = 0.01
        self.paddle_y = 0.5
