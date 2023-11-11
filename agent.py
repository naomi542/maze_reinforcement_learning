import random

class Agent:
    def __init__(self, env, learning_rate=0.1, epsilon=0.05, discount_factor=1):
        """
        Parameters:
        -----------
        learning_rate: float, learning rate used in Q-learning update step
        epsilon: float, between [0, 1), probability that the agent will take a random action
        discount_factor: float, by default 1, the importance of future rewards 
        """
        self.env = env

        # for each cell, create a dictionary containing the q value for each move allowed from that cell
        q_values = []
        for i in range(env.n_cells):
            cell_actions_q = {}
            for action in env.cell_actions[i]:
                cell_actions_q[action] = 0
            q_values.append(cell_actions_q)

        self.q_values = q_values
        self.lr = learning_rate
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.td_history = []
        self.q_history = {}

    def get_action(self, observation):
        """Epsilon-greedy action selection

        Parameters:
        -----------
        observation: int, observation of the agent's state from the environment

        Returns:
        --------
        action: Action, selected action the agent will perform
        """
        # epsilon greedy sampling action sampling
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.sample_actions()
        else:
            action = max(self.q_values[observation], key=self.q_values[observation].get) 
        return action
        
    def Q_update(self, observation, action, reward, next_observation):
        """Updates the Q values after taking an action in the environment

        Parameters:
        -----------
        observation: int, obesrvation of the agent's state from the environment
        action: Action, action the agent selected to perform
        reward: float, the reward of performing the action from the environment
        next_observation: int, the next observation the agent will receive after performing its action
        """
        # Compute the TD loss for Q-learing
        future_q_value = max(self.q_values[next_observation].values())
        td_loss = reward + self.discount_factor * future_q_value - self.q_values[observation][action]
        # Write the update rule for Q(s, a)
        self.q_values[observation][action] += self.lr * td_loss
        self.td_history.append(td_loss)
                
    def SARSA_update(self, observation, action, reward, next_observation, next_action):
        """Updates the Q values after taking an action in the environment using SARSA

        Parameters:
        -----------
        observation: int, obesrvation of the agent's state from the environment
        action: Action, action the agent selected to perform
        reward: float, the reward of performing the action from the environment
        next_observation: int, the next observation the agent will receive after performing its action
        """
        td_loss = reward + ((self.discount_factor * self.q_values[next_observation][next_action]) - (self.q_values[observation][action]))
        self.q_values[observation][action] += self.lr*td_loss
        self.td_history.append(td_loss)
        
        
        
        

