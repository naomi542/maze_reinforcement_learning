from enum import Enum
import numpy as np
from random import sample
import pandas as pd

class Action(Enum):
    MOVE_LEFT = [-1, 0]
    MOVE_RIGHT = [1, 0]
    MOVE_UP = [0, -1]
    MOVE_DOWN = [0, 1]

class Turn(Enum):
    RIGHT_TURN = -1
    LEFT_TURN = 1

class Status(Enum):
    SEARCHING = 0
    WATER_REACHED = 1

class Environment:
    def __init__(self, maze) -> None:
        self.maze = maze
        self.metrics = None
        self.cell_actions = self.generate_cell_actions()
        self.n_cells = len(self.cell_actions)
        self.reset()

    def update_metrics(self, action, diff):
        self.metrics['Total # of actions'] += 1

        # check if dead end
        if len(self.cell_actions[self.current_cell]) > 1:
            self.metrics['# of actions w/o dead ends'] += 1

        # corridor
        if self.current_cell not in self.nodes:
            self.metrics['# of corridor actions'] += 1

            if self.last_action == action:
                self.metrics['corridor actions']['straight'] += 1
            else:
                self.metrics['corridor actions']['turn around'] += 1
        # node
        else:
            self.metrics['# of node actions'] += 1

            if diff == Turn.LEFT_TURN.value:
                self.metrics['node actions']['left turn'] += 1
            elif diff == Turn.RIGHT_TURN.value:
                self.metrics['node actions']['right turn'] += 1
            elif diff == 0:
                self.metrics['node actions']['straight through node'] += 1
            else:
                self.metrics['node actions']['node turn around'] += 1

        # consecutive turns
        if self.last_turn.value == diff:
            if self.consecutive_turn_count not in self.metrics['consecutive turns']:
                self.metrics['consecutive turns'][self.consecutive_turn_count]  = 1
            else:
                self.metrics['consecutive turns'][self.consecutive_turn_count] += 1

            if self.consecutive_turn_count > 1:
                self.metrics['consecutive turns'][self.consecutive_turn_count - 1] -= 1

        # unique nodes
        self.metrics['total # of unique nodes visisted'] = len(self.visited_nodes)
        self.metrics["total # of unique nodes visisted before water"] = len(self.visited_nodes)

        # node visites
        if self.current_cell in self.nodes:
            self.metrics["# of visits for each visited node"][self.nodes.index(self.current_cell)] += 1
            self.metrics["# of visits for each visited node before water"][self.nodes.index(self.current_cell)] += 1


        # "% of each corridor action"
        total_corridor = 0
        for key, value in self.metrics['corridor actions'].items():
            total_corridor += value
        if total_corridor > 0:
            for key, value in self.metrics['corridor actions'].items():
                self.metrics["% of each corridor action"][key] = round(value * 100 / total_corridor, 2)

        # % of each node action
        if self.metrics['# of node actions'] > 0:
            for key, value in self.metrics['node actions'].items():
                self.metrics['% of each node action'][key] = round(value * 100 / self.metrics['# of node actions'], 2)

        # % of consecutive turn intervals
        total_consec_turns = 0
        for key, value in self.metrics['consecutive turns'].items():
            total_consec_turns += value
        if total_consec_turns > 0:
            for key, value in self.metrics['consecutive turns'].items():
                self.metrics["% of consecutive turn intervals"][key] = round(value * 100 / total_consec_turns, 2)

    def reset(self):
        """Reset the environment"""
        self.current_cell = 0
        self.visited_cells = set()
        self.visited_nodes = set()
        self.last_turn = None
        self.last_action = Action.MOVE_RIGHT
        self.current_run = 0 # counts number of times current action has been taken
        self.status = Status.SEARCHING
        self.nodes = [x[-1] for x in self.maze.ru]
        self.consecutive_turn_count = 0
        self.metrics = {'Total # of actions': 0,
                        '# of actions w/o dead ends': 0,
                        '# of corridor actions': 0, 
                        '# of node actions': 0,
                        'corridor actions': {'straight': 0, 'turn around': 0},
                        'node actions': {'left turn': 0, 'right turn': 0, 'straight through node': 0, 'node turn around': 0},
                        'consecutive turns': {},
                        "% of each corridor action": {}, 
                        "% of each node action": {},
                        "% of consecutive turn intervals": {}, 
                        "total # of unique nodes visisted": 0,
                        "# of visits for each visited node": np.zeros(len(self.nodes)), 
                        "water_reached": True,
                        "total # of unique nodes visisted before water": 0,
                        "# of visits for each visited node before water": np.zeros(len(self.nodes))}
        return self.current_cell
    
    def step(self, action):
        """Take a step with the specified action"""
        if action not in self.cell_actions[self.current_cell]:
            print("Move not allowed!")
            return
        
        if self.current_cell in self.nodes:
            self.visited_nodes.add(self.current_cell)
        
        reward = self.get_reward(action)


        self.visited_cells.add(self.current_cell)
        self.last_action = action
        return self.current_cell, reward, self.status
    
    def get_reward(self, action):
        """determine the reward of taking the given action"""
        reward = 0
        if self.last_turn is not None:
            moves_dict = {Action.MOVE_RIGHT: 1, Action.MOVE_DOWN: 2, Action.MOVE_LEFT: 3, Action.MOVE_UP: 4}

            diff = moves_dict[self.last_action] % len(moves_dict) - moves_dict[action] % len(moves_dict)

            self.update_metrics(action, diff)

            # same as last turn, negative reward
            if diff == self.last_turn.value:
                reward = -0.1
                self.consecutive_turn_count +=1
            elif self.current_cell in self.nodes and diff != self.last_turn.value:
                self.consecutive_turn_count = 0
            
            if diff in [-1, 1]:
                self.last_turn = Turn(diff)
        else:
            # right turn at the start
            if action == Action.MOVE_DOWN:
                self.last_turn = Turn.RIGHT_TURN
            # left turn
            elif action == Action.MOVE_UP:
                self.last_turn = Turn.LEFT_TURN

        if self.current_cell in self.nodes and len(self.cell_actions[self.current_cell]) > 1 and action == self.last_action:
            # penalize moving ahead instead of making a turn at a junction
            reward = -0.1

        next_coord = np.add(action.value, [self.maze.xc[self.current_cell], self.maze.yc[self.current_cell]])
        self.current_cell = self.maze.ce[tuple(next_coord)]

        if self.current_cell == 165: # reached water
            reward = 10
            self.status = Status.WATER_REACHED
        elif self.current_cell in self.visited_cells: # penalty for going to a cell already visited
            reward = -0.25

        return reward

    def sample_actions(self):
        """randomly sample an action from allowable actions for current cell"""
        return sample(self.cell_actions[self.current_cell], 1)[0]
    
    def generate_cell_actions(self):
        """Returns a list of lists, where each cell has a list of allowable actions"""
        cell_actions = []
        for i in range(len(self.maze.xc)):
            cell_actions.append(self.get_actions_allowed(i))
        return cell_actions
        
    def get_actions_allowed(self, cell_idx):
        """Returns the moves allowed from the current cell based on the maze walls and boundaries."""
        x = self.maze.xc[cell_idx]
        y = self.maze.yc[cell_idx]

        moves = []
        if [x, y - 0.5] not in self.maze.wall_midpoints and y - 1 >= 0:
            moves.append(Action.MOVE_UP)
        if [x, y + 0.5] not in self.maze.wall_midpoints and y + 1 <= 14:
            moves.append(Action.MOVE_DOWN)
        if [x - 0.5, y] not in self.maze.wall_midpoints and x - 1 >= 0:
            moves.append(Action.MOVE_LEFT)
        if [x + 0.5, y] not in self.maze.wall_midpoints and x + 1 <= 14:
            moves.append(Action.MOVE_RIGHT)

        return moves
    
    
    def convert_to_df(self, episode):
        
        dictionary = self.metrics
        
        dictionary['# of visits for each visited node'] = pd.Series(dictionary['# of visits for each visited node'])
        dictionary['# of visits for each visited node before water'] = pd.Series(dictionary['# of visits for each visited node before water'])
        
        # Convert entire dictionary into dataframe
        df = pd.DataFrame.from_dict(dictionary, orient='index').T
        df = df.rename(index={0: episode}, inplace=False)

        return df