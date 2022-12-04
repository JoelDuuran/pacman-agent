# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint
from capture import GameState
from problems import *

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='MyAttackAgent', second='MyDefenseAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state:GameState):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state:GameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state:GameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state:GameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

class MyDefenseAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.en_detected = False
        self.in_frontier = False
        self.food_eaten = False
        self.last_eaten_food = None
        self.old_food = None
        self.my_food = None
    
    def choose_action(self, game_state:GameState):
        DevourProblem = DevourSearchProblem(self.index, game_state, self.get_maze_distance)
        IntimidateProblem = IntimidateSearchProblem(self.index, game_state)
        FrontProblem = FrontSearchProblem(self.index, game_state)
        current_position = game_state.get_agent_position(self.index)
        if not self.old_food: self.old_food = self.get_food_you_are_defending(game_state)
        self.my_food = self.get_food_you_are_defending(game_state)

        self.en_detected = False
        self.ongoing = False

        DevourProblem.en_pos = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
        # for all enemies near
        for en_pos in DevourProblem.en_pos:
            #if there is any
            if en_pos is not None:
                #if he is on our team's field then en_detected=True
                if (en_pos[0] < 16 and game_state.is_on_red_team(self.index) or en_pos[0] > 15 and not game_state.is_on_red_team(self.index)):
                    self.en_detected = True
                


        if self.last_eaten_food and self.last_eaten_food == current_position:
            self.food_eaten = False

        for i, food_row in enumerate(self.my_food):
            for j, food in enumerate(food_row):
                if food != self.old_food[i][j]:
                    self.food_eaten = True
                    self.last_eaten_food = (i, j)
                    self.old_food = self.get_food_you_are_defending(game_state)        
         
        
        if self.en_detected:
            path = aStarSearch(DevourProblem, DefenseManhattanHeuristic)
            return path[0]
        elif self.food_eaten:
            EatenFoodProblem = EatenFoodSearchProblem(self.index, game_state, self.last_eaten_food)
            path = aStarSearch(EatenFoodProblem, EatenFoodManhattanHeuristic)

            return path[0]
        else:
            path = aStarSearch(IntimidateProblem, IntimidateManhattanHeuristic)
            if len(path) == 0:
                return Directions.STOP
            return path[0]



class MyAttackAgent(CaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)
        self.retrieve = False
        self.food = None
        self.capsules = None
        self.eaten = 0
        self.en_detected = False
        self.capsuleTime = 0
        self.camper = 0
        self.side = 2
        self.trying = 0
        self.camp_detected = False

    def choose_action(self, game_state:GameState):
        
        FoodSearchProblem = AnyFoodSearchProblem(self.index, game_state)
        ReturnProblem = ReturnSearchProblem(self.index, game_state)
        EscapeProblem = EscapeSearchProblem(self.index, game_state, self.get_maze_distance)
        DevourProblem = DevourSearchProblem(self.index, game_state, self.get_maze_distance)
        AnyPillProblem = AnyPillSearchProblem(self.index, game_state)
        CamperProblem1 = ChangeEntranceProblem1(self.index, game_state)
        CamperProblem2= ChangeEntranceProblem2(self.index, game_state)
        CamperProblem3 = ChangeEntranceProblem3(self.index, game_state)
        current_position = game_state.get_agent_position(self.index)
        is_red = game_state.is_on_red_team(self.index)
        in_base = ReturnProblem.is_in_base(current_position)
        #If pacman died, reset the number of food eaten
        if game_state.get_initial_agent_position(self.index) == current_position:
            self.eaten = 0
        
        #If in base then poits are already retrieved
        if not game_state.data.agent_states[self.index].is_pacman:
            self.eaten = 0
        if in_base:
            self.retrieve = False
            self.en_detected = False
            self.camp_detected = False
            
            DevourProblem.en_pos = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
            attacker_en = [pos for pos in DevourProblem.en_pos if pos and self.get_maze_distance(current_position, pos) <= 8]
            # for all enemies near
            
            for en_pos in DevourProblem.en_pos:
                #if there is any
                if en_pos is not None:
                    #if he is on our team's field then en_detected=True
                    if (en_pos[0] < 16 and game_state.is_on_red_team(self.index) or en_pos[0] > 15 and not game_state.is_on_red_team(self.index)):
                        self.en_detected = True
                    
            for en_pos in attacker_en:
                if (en_pos[en_pos[0] >= 16 and game_state.is_on_red_team(self.index) or en_pos[0] <= 15 and not game_state.is_on_red_team(self.index)]):
                        self.camp_detected = True
            if self.camp_detected:
                self.camper += 1
            if self.camper >= 5:
                if self.side == 0:

                    if not CamperProblem1.isGoalState(current_position):
                        path = aStarSearch(CamperProblem1, MoveManhattanHeuristic1)
                        self.trying += 1
                        if self.trying >= 10:
                            self.camper = 0
                            self.side = 1
                        return path[0]
                    else:
                        self.side = 1
                        self.camper = 0
                        self.trying = 0
                elif self.side == 1:
                    if not CamperProblem2.isGoalState(current_position):
                        
                        self.trying += 1
                        if self.trying >= 10:
                            self.camper = 0
                            self.side = 2
                            self.trying = 0
                        path = aStarSearch(CamperProblem2, MoveManhattanHeuristic2)
                        return path[0]
                    else:
                        self.side = 2
                        self.camper = 0
                        self.trying = 0
                elif self.side == 2:
                    if not CamperProblem3.isGoalState(current_position):
                        self.trying += 1
                        if self.trying >= 10:
                            self.camper = 0
                            self.side = 0
                            self.trying = 0
                        
                        path = aStarSearch(CamperProblem3, MoveManhattanHeuristic3)
                        return path[0]
                    else:
                        self.trying = 0
                        self.side = 0
                        self.camper = 0

                
            if self.en_detected and min([self.get_maze_distance(current_position, pos) for pos in DevourProblem.en_pos if pos])<=7:
                path = aStarSearch(DevourProblem, DefenseManhattanHeuristic)
                return path[0]


        #Add 1 when food eaten
        if self.food and current_position in self.food.as_list(): 
            self.eaten += 1
        
        
        if self.capsules and current_position in self.capsules:
            self.capsuleTime = 40
        if self.capsuleTime > 0 and game_state.data.agent_states[self.index].is_pacman:
            self.capsuleTime -= 1
        #When 3 food or more are eaten, come back to the base
        if self.eaten >= 3: 
            
            #Check if there are close foods that could be eaten without risk
            x, y = current_position
            adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), (x+1,y+1), (x+1,y-1), (x-1,y+1), (x-1,y-1 )]
            close_food = False
            for cell in adjacent:
                if cell in self.food.as_list():
                    close_food = True
            
            if not close_food: 
                #If there is no more close food, retrieve what you have
                self.retrieve = True
        
        self.food = game_state.get_blue_food() if is_red else game_state.get_red_food()
        self.capsules = game_state.get_blue_capsules() if is_red else game_state.get_red_capsules()

        #Checks if it detects an enemy
        self.en_detected = False

        temp = [game_state.get_agent_position(i) for i in self.get_opponents(game_state)]
        EscapeProblem.en_pos = [ghost_pos for ghost_pos in temp if ghost_pos and self.get_maze_distance(current_position, ghost_pos) <= 5]
        for en_pos in EscapeProblem.en_pos:
            if en_pos is not None and not ReturnProblem.is_in_base(en_pos):
                    self.en_detected = True
                    

        ################Search########################
    
        #Default go search for some food
        if not self.retrieve:
            if self.en_detected:
                
                if self.capsules:
                    dist_to_pill = self.get_maze_distance(current_position, self.capsules[0])
                    en_dist_to_pill = min([self.get_maze_distance(enemy, self.capsules[0]) for enemy in EscapeProblem.en_pos])
                    if (dist_to_pill < en_dist_to_pill):
                        path = aStarSearch(AnyPillProblem, PillManhattanHeuristic)
                        if path:
                            return path[0]
                        else:
                            return Directions.STOP
                if self.capsuleTime > 3:
                    
                    path = aStarSearch(FoodSearchProblem, FoodManhattanHeuristic)
                    if path:
                        return path[0]
                    else:
                        return Directions.STOP
                
                path = breadthFirstSearch(EscapeProblem)
                if path:
                    return path[0]
                else:
                    return Directions.STOP
            else:
                path = aStarSearch(FoodSearchProblem, FoodManhattanHeuristic)
                #print(f"The current path to closest is: {path}")
                if path:
                    return path[0]
                else:
                    return Directions.STOP

        #When some food has been eaten, come back to the base
        
        if self.retrieve:        
            if self.en_detected:
                if self.capsuleTime > 3:
                    x, y = current_position
                    adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                    close_food = False
                    for cell in adjacent:
                        if cell in self.food.as_list(): 
                            #The action to go to that cell           
                            return Actions.vector_to_direction((cell[0] - current_position[0], cell[1] - current_position[1]))      
                    path = aStarSearch(ReturnProblem, ReturnManhattanHeuristic)
                    if path:
                        return path[0]
                    else:
                        return Directions.STOP
                
                path = breadthFirstSearch(EscapeProblem)
                if path:
                    return path[0]
                else:
                    return Directions.STOP
            else:
                x, y = current_position
                adjacent = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
                close_food = False
                for cell in adjacent:
                    if cell in self.food.as_list(): 
                        #The action to go to that cell
                        
                        return Actions.vector_to_direction((cell[0] - current_position[0], cell[1] - current_position[1]))
                    
                path = aStarSearch(ReturnProblem, ReturnManhattanHeuristic)
                if path:
                    return path[0]
                else:
                    return Directions.STOP



def nullHeuristic(state, problem=None):
    return 0

def DefenseManhattanHeuristic(state:tuple[int, int], problem:DevourSearchProblem):
    return min([problem.mazeDistance(state, it_pos) for it_pos in problem.en_pos if it_pos ])

def IntimidateManhattanHeuristic(state:tuple[int, int], problem:IntimidateSearchProblem):
    return min(abs(12-state[0]), abs(19-state[0]))

def EscapeManhattanHeuristic(state:tuple[int, int], problem:EscapeSearchProblem):
    return (5 - DefenseManhattanHeuristic(state, problem))

def FoodManhattanHeuristic(state:tuple[int, int], problem:AnyFoodSearchProblem=None, info={}):    
    return min([util.manhattanDistance(state, xy2) for xy2 in problem.food])

def MoveManhattanHeuristic1(state:tuple[int, int], problem:ChangeEntranceProblem1):
    return abs(1.5-state[1])+abs(15.5-state[0])



def MoveManhattanHeuristic2(state:tuple[int, int], problem:ChangeEntranceProblem2):
    return abs(7.5-state[1])+abs(15.5-state[0])

def MoveManhattanHeuristic3(state:tuple[int, int], problem:ChangeEntranceProblem3):
    return abs(13.5-state[1])+abs(15.5-state[0])

def PillManhattanHeuristic(state:tuple[int, int], problem:AnyPillSearchProblem=None, info={}):    
    return min([util.manhattanDistance(state, xy2) for xy2 in problem.capsules])

def EatenFoodManhattanHeuristic(state, problem:EatenFoodSearchProblem):
    return util.manhattanDistance(state, problem.eaten_food)

def ReturnManhattanHeuristic(state:tuple[int, int], problem:ReturnSearchProblem=None):
    return abs(15.5-state[0])

def aStarSearch(problem:AnyFoodSearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    start_pos = problem.getStartState() #We get a tuple (GameState, food)
    
    current_node = start_pos, [], 0 #A node has the state, the Actions taken to arrive at such state and the cost
    visited = []
    frontier = util.PriorityQueue()
    frontier.push(current_node, current_node[2] + heuristic(start_pos, problem))
    #Loop while there is still nodes to visit
    while(frontier.isEmpty() == False):
        #Visit the node on top of the stack and add it to the visited nodes
        while(current_node[0] in visited):
            if frontier.isEmpty == True:
                return None
            current_node = frontier.pop()
        visited.append(current_node[0])


        #End of the loop when Goal State achieved, we return the moves 
        if(problem.isGoalState(current_node[0]) and current_node[1]):
            return current_node[1]

        #Find all the successor states and the moves to achieve them
        children = problem.getSuccessors(current_node[0])
        #Loop through all the possible successors
        for (pos,move,cost) in children:

            if(pos not in visited): 

                new_node = (pos, current_node[1] + [move], current_node[2] + cost)
                
                f_value = new_node[2] + heuristic(pos, problem)
               
                frontier.update(new_node, f_value)

    

def breadthFirstSearch(problem:AnyFoodSearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
#We initialize the data structures that we will need for the search. We will need to store the moves in the nodes 
    # as we will need to return them at the end
    current_node = (problem.getStartState(), [])
    visited = []
    frontier = util.Queue()
    frontier.push(current_node)

    #Loop while there is still nodes to visit
    while(frontier.isEmpty() == False):

        #Visit the node on top of the stack and add it to the visited nodes
        current_node = frontier.pop()
        visited.append(current_node[0])


        #End of the loop when Goal State achieved, we return the moves 
        if(problem.isGoalState(current_node[0])):
            return current_node[1]


        #Find all the successor states and the moves to achieve them
        children = problem.getSuccessors(current_node[0])

        #Loop through all the possible successors
        for (state,move , _) in children:

            #If they are not yet in the visited or frontier, add the new node to the frontier adding the move 
            # to the list of already done moves

            in_frontier = False
            for (current_state, _) in frontier.list :
                if(state == current_state): 
                    in_frontier = True

            if( not in_frontier and state not in visited): 
                #print(current_node[1])
                new_node = (state, current_node[1] + [move])
                frontier.push(new_node)
            
    
    util.raiseNotDefined()  