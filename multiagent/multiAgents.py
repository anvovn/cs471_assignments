# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()
        if successorGameState.isWin() or successorGameState.isLose():
            return score

        # Food scoring
        foodList = newFood.asList()
        if foodList:
            foodDistances = [manhattanDistance(newPos, food) for food in foodList]
            score += 10.0 / min(foodDistances)  # Closer food increases score
            score -= 2.0 * len(foodList)

        # Ghost scoring
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostDistance = manhattanDistance(newPos, ghostState.getPosition())

            if scaredTime > 0:
                if ghostDistance < scaredTime:
                    score += 200.0 / (ghostDistance + 1)  # Closer scared ghost increases score
                elif ghostDistance <= 1:
                    score -= 1000
            else:
                score -= 20.0 / (ghostDistance * ghostDistance)  # Closer active ghost decreases score

        # Determine if there are ghosts nearby with proximity to a capsule
        capsules = successorGameState.getCapsules()
        dangerousGhostNearby = any(
            scaredTime == 0 
            and manhattanDistance(newPos, ghostState.getPosition()) < 5
            for ghostState, scaredTime in zip(newGhostStates, newScaredTimes)
        )    
        if capsules and dangerousGhostNearby:
            capsuleDistances = [manhattanDistance(newPos, capsule) for capsule in capsules]
            score += 50.0 / (min(capsuleDistances) + 1) # Closer capsule increases score

        if action == Directions.STOP:
            score -= 5  # Discourage stopping

        return score

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, bestAction = self.minimax(gameState, 0, 0)
        return bestAction

    def minimax(self, state: GameState, depth: int, agentIndex: int):

        # Check win/loss, depth and move conditions
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state), Directions.STOP

        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state), Directions.STOP

        # Determine next agent and increase depth when reaching back to Pacman
        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextValue = depth + 1 if nextAgent == 0 else depth

        bestAction = actions[0]
        bestValue = float('-inf') if agentIndex == 0 else float('inf')

        # Recursive search on successor states
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            value, _ = self.minimax(successor, nextValue, nextAgent)

            if agentIndex == 0:  # Pacman's turn (maximizing)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            else:  # Ghost's turn (minimizing)
                if value < bestValue:
                    bestValue = value
                    bestAction = action
        return bestValue, bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        _, bestAction = self.alphaBeta(
            gameState, 0, 0, float("-inf"), float("inf")
        )
        return bestAction

    def alphaBeta(self, state, depth, agentIndex, alpha, beta):

        if state.isWin() or state.isLose() or depth >= self.depth:
            return self.evaluationFunction(state), Directions.STOP

        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state), Directions.STOP

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        bestAction = actions[0]
        bestValue = float("-inf") if agentIndex == 0 else float("inf")

        # Recursive search on successor states
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            value, _ = self.alphaBeta(
                successor, nextDepth, nextAgent, alpha, beta
            )

            if agentIndex == 0: # Max, alpha calculations
                if value > bestValue:
                    bestValue = value
                    bestAction = action

                if bestValue > beta:
                    return bestValue, bestAction

                alpha = max(alpha, bestValue)

            else: # Min, beta calculations
                if value < bestValue:
                    bestValue = value
                    bestAction = action

                if bestValue < alpha:
                    return bestValue, bestAction

                beta = min(beta, bestValue)

        return bestValue, bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, bestAction = self.expectimax(gameState, 0, 0)
        return bestAction

    def expectimax(self, state: GameState, depth: int, agentIndex: int):

        if state.isWin() or state.isLose() or depth >= self.depth:
            return self.evaluationFunction(state), Directions.STOP

        actions = state.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(state), Directions.STOP

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        nextDepth = depth + 1 if nextAgent == 0 else depth

        bestValue = float("-inf")
        bestAction = actions[0]
        totalValue = 0.0 # Sum of ghost action values

        # Recursive search on successor states
        for action in actions:
            successor = state.generateSuccessor(agentIndex, action)
            value, _ = self.expectimax(successor, nextDepth, nextAgent)

            if agentIndex == 0: # Max
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            else: # Chance
                totalValue += value

        if agentIndex == 0: 
            return bestValue, bestAction
        else: # Chance calculation
            return totalValue / len(actions), Directions.STOP


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Since it is to be used with the search agents, it will evaluate the states as opposed to actions
    like in the reflex agent. This implementation uses the same scoring as the reflex agent's evaluation function in terms
    of determining pacman's positions relative to food, ghost, and capsule positions where closer distances between them
    either increase or decrease the score. The main difference is that there's no successor generation as that is done
    through the search, therefore this function only needs to evaluate the state given to it. There is also no stopping
    penalty like the reflex agent.  
    """

    # Game State Information
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    score = currentGameState.getScore()
    if currentGameState.isWin() or currentGameState.isLose():
        return score

    foodList = food.asList()
    if foodList:
        foodDistances = [manhattanDistance(position, food) for food in foodList]
        score += 10.0 / min(foodDistances)  # Closer food increases score
        score -= 2.0 * len(foodList)

    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        ghostDistance = manhattanDistance(position, ghostState.getPosition())

        if scaredTime > 0:
            if ghostDistance < scaredTime:
                score += 200.0 / (ghostDistance + 1)  # Closer scared ghost increases score
            elif ghostDistance <= 1:
                score -= 1000
        else:
            score -= 20.0 / (ghostDistance * ghostDistance)  # Closer active ghost decreases score

    capsules = currentGameState.getCapsules()
    dangerousGhostNearby = any(
        scaredTime == 0
        and manhattanDistance(position, ghostState.getPosition()) < 5
        for ghostState, scaredTime in zip(ghostStates, scaredTimes)
    )    
    if capsules and dangerousGhostNearby:
        capsuleDistances = [manhattanDistance(position, capsule) for capsule in capsules]
        score += 50.0 / (min(capsuleDistances) + 1) # Closer capsule increases score

    return score

# Abbreviation
better = betterEvaluationFunction
