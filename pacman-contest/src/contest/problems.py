import util
from game import Actions, Directions
from capture import GameState


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

class PositionSearchProblem(SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, index, gameState:GameState, costFn, goal, start=None):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.index = index



    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        return state == self.goal


    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.direction_to_vector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, index, gameState:GameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.get_blue_food() if gameState.is_on_red_team(index) else gameState.get_red_food()
        self.index = index
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        food = self.food.as_list() #Convert the food grill into a list of coordinates of the remaining food

        "*** YOUR CODE HERE ***"
        return state in food 

class AnyPillSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, index, gameState:GameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.capsules = gameState.get_blue_capsules() if gameState.is_on_red_team(index) else gameState.get_red_capsules()
        self.index = index
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        capsules = self.capsules #Convert the food grill into a list of coordinates of the remaining food

        "*** YOUR CODE HERE ***"
        return state in capsules 

class ReturnSearchProblem(PositionSearchProblem):
    def __init__(self, index, gameState:GameState) -> None:
        self.is_in_base = (lambda x: x[0] < 16) if gameState.is_on_red_team(index) else (lambda x: x[0] >= 16)
        self.index = index
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        return self.is_in_base(state)

class EscapeSearchProblem(PositionSearchProblem):
    def __init__(self, index, gameState: GameState, dist):
        self.is_in_base = (lambda x: x[0] < 16) if gameState.is_on_red_team(index) else (lambda x: x[0] >= 16)
        self.index = index
        self.mazeDist = dist
        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1
        self.en_pos = []
        

    def isGoalState(self, state):
        return min([self.mazeDist(state, en_pos) for en_pos in self.en_pos if en_pos]) > 8

class DevourSearchProblem(PositionSearchProblem):
    '''This problem consists pursuing a pacman while it is in it's territory'''
    def __init__(self, index, gameState: GameState, dist):
        self.is_in_base = (lambda x: x[0] < 16) if gameState.is_on_red_team(index) else (lambda x: x[0] >= 16)
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1
        self.mazeDistance = dist
        self.en_pos = []

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        successors = []
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        for action in actions:
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)

                successors.append( ( nextState, action, cost) )


        return successors

    def isGoalState(self, state:tuple[int, int]):
        return state in self.en_pos
    
class FrontSearchProblem(PositionSearchProblem):
    '''This problem consists in getting close to the frontier'''
    def __init__(self, index, gameState: GameState):
        self.is_in_frontier = (lambda x: x[0] <= 15 and x[0] >= 13 and x[1] >= 7 and x[1] <= 8) if gameState.is_on_red_team(index) else (lambda x: x[0] >= 16 and x[0] <= 18 and x[1] <= 8 and x[1] >= 7)
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        return self.is_in_frontier(state)
    
class ChangeEntranceProblem1(PositionSearchProblem):
    '''This problem consists in moving when a defender is defending constantly the same place'''
    def __init__(self, index, gameState: GameState):
        self.moved = (lambda x: x[0] == 15 and x[1] >= 1 and x[1] <= 2) if gameState.is_on_red_team(index) else (lambda x: x[0] == 16 and x[1] <= 2 and x[1] >= 1)
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        return self.moved(state)

class ChangeEntranceProblem2(PositionSearchProblem):
    '''This problem consists in moving when a defender is defending constantly the same place'''
    def __init__(self, index, gameState: GameState):
        self.moved = (lambda x: x[0] == 15 and x[1] >= 7 and x[1] <= 8) if gameState.is_on_red_team(index) else (lambda x: x[0] == 16 and x[1] <= 8 and x[1] >= 7)
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        return self.moved(state)

class ChangeEntranceProblem3(PositionSearchProblem):
    '''This problem consists in moving when a defender is defending constantly the same place'''
    def __init__(self, index, gameState: GameState):
        self.moved = (lambda x: x[0] == 15 and x[1] >= 13 and x[1] <= 14) if gameState.is_on_red_team(index) else (lambda x: x[0] == 16 and x[1] <= 14 and x[1] >= 13)
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        return self.moved(state) 

class IntimidateSearchProblem(PositionSearchProblem):
    '''This problem consists in staying at the frontier to intimidate possible invaders and '''
    def __init__(self, index, gameState: GameState):
        self.is_covering = (lambda x: x[0] == 15 and x[1] >= 7 and x[1] <= 8) if gameState.is_on_red_team(index) else (lambda x: x[0] == 16 and x[1] <= 8 and x[1] >= 7)
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1

    def isGoalState(self, state):
        return self.is_covering(state)

class EatenFoodSearchProblem(PositionSearchProblem):
    def __init__(self, index, gameState: GameState, lastEaten:tuple[int, int]):
        self.index = index
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_position(index)
        self.costFn = lambda x: 1
        self.eaten_food = lastEaten

    def isGoalState(self, state):
        print(state == self.eaten_food)
        return state == self.eaten_food 