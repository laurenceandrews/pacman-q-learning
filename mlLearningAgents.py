# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util

from collections import defaultdict
import math

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        self.pacmanPos = state.getPacmanPosition()
        self.ghostPositions = state.getGhostPositions()
        self.food = state.getFood().asList()
        # util.raiseNotDefined()


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 5,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        self.qValues = {}
        self.qValue = util.Counter()
        self.counts = {}
        self.prevState = []
        self.prevAction = []
        self.qScore = 0

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts
    
    def qLearn(self, state):
        # Get the list of legal actions Pacman can take in the given state
        availableActions = state.getLegalPacmanActions()

        # If Pacman has taken less than half of the training episodes and STOP action is legal, remove it from availableActions
        if self.getEpisodesSoFar() / self.getNumTraining() < 0.5 and Directions.STOP in availableActions:
            availableActions.remove(Directions.STOP)
        
        # Initialize a counter to hold the Q-value estimates for each available action
        qValues = util.Counter()
        
        # For each available action, retrieve the Q-value estimate for the current state-action pair (state, action), or 0 if it doesn't exist yet
        for action in availableActions:
            qValues[action] = self.qValue.get((state, action), 0)

        # If there is a previous action in the episode, check if it's safe to reverse the action and remove it from availableActions if not
        if self.prevAction:
            prevAction = self.prevAction[-1]
            distanceX = state.getPacmanPosition()[0] - state.getGhostPosition(1)[0]
            distanceY = state.getPacmanPosition()[1] - state.getGhostPosition(1)[1]
            if math.sqrt(distanceX ** 2 + distanceY ** 2) > 2 and Directions.REVERSE[prevAction] in availableActions and len(availableActions) > 1:
                availableActions.remove(Directions.REVERSE[prevAction])
        
        # Return the action with the highest Q-value estimate according to qValues
        return qValues.argMax()

    def qUpdate(self, state, action, reward, qMax):
        # Retrieve the old Q-value for the state-action pair, or 0 if it does not exist yet
        oldQ = self.qValue.get((state, action), 0)
        # Calculate the new Q-value based on the old Q-value, the reward, the learning rate (alpha), and the discount factor (gamma)
        newQ = oldQ + self.alpha*(reward + self.gamma * qMax - oldQ)
        # Update the Q-value for the state-action pair in the Q-value table
        self.qValue[(state, action)] = newQ

    def qMax(self, state):
        # Get the list of Q-values for all possible legal actions
        qList = [self.qValue[(state, action)] for action in state.getLegalPacmanActions()]
        # Return the maximum Q-value or 0 if there are no legal actions
        return max(qList, default=0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        score = endState.getScore()
        return score - startState.getScore()
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        # Convert the game state to a tuple
        stateTuple = (state.pacmanPos, tuple(state.ghostPositions), tuple(state.food))
        # Check if the state has not been visited before
        if stateTuple not in self.qValues:
            return 0.0
        # Check if the action has not been taken in the state before
        if action not in self.qValues[stateTuple]:
            return 0.0
        # Return the Q-value for the given state and action
        return self.qValues[stateTuple][action]
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        # Get the list of legal actions
        legal = state.getLegalPacmanActions()

        # If stopping is a legal action, remove it from the list of legal actions
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Create a tuple representing the current state of the game
        stateTuple = (state.pacmanPos, tuple(state.ghostPositions), tuple(state.food))

        # If the state has not been previously encountered, return a default value of 0.0
        if stateTuple not in self.qValues:
            return 0.0

        # Return the maximum Q-value for the legal actions in the current state
        # by iterating over each action in legal and getting its Q-value from self.qValues
        # and taking the max of the Q-values
        return max([self.qValues[stateTuple][action] for action in legal])
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Get the Q-value of the current state-action pair
        oldQValue = self.getQValue(state, action)
        # Compute the sample using the reward and the maximum Q-value of the next state
        sample = reward + self.gamma * self.maxQValue(nextState)
        # Compute the new Q-value using the learning rate and the old and new samples
        newQValue = (1 - self.alpha) * oldQValue + self.alpha * sample
        # Update the Q-value of the current state-action pair in the Q-value dictionary
        self.qValues[(state, action)] = newQValue
        # Update the state-action pair count for the current state
        self.updateCount(state, action)
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.counts.setdefault(state, defaultdict(int))
        self.counts[state][action] += 1
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts[state][action]
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts == 0:
            return float("inf")
        return utility + self.epsilon * pow(counts, -0.5)
        # util.raiseNotDefined()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        # print("Legal moves: ", legal)
        # print("Pacman position: ", state.getPacmanPosition())
        # print("Ghost positions:", state.getGhostPositions())
        # print("Food locations: ")
        # print(state.getFood())
        # print("Score: ", state.getScore())

        # Calculate the difference between the current score and the previous score in qScore
        scoreDiff = state.getScore() - self.qScore

        # If there is a previous state, retrieve it along with the previous action taken and update the Q-values using qUpdate()
        if len(self.prevState) != 0:
            prevState, prevAction = self.prevState[-1], self.prevAction[-1]
            maxQScore = self.qMax(state)
            self.qUpdate(prevState, prevAction, scoreDiff, maxQScore)

        # Retrieve the legal actions available in the current state
        legal = state.getLegalPacmanActions()

        # Determine whether to explore or exploit by generating a random number and comparing it with the exploration rate
        if random.random() > self.epsilon:
            # If it is time to exploit, determine the best action to take using qLearn()
            move = self.qLearn(state)
        else:
            # If it is time to explore, randomly select an action from the legal actions available
            move = random.choice(legal)

        # Record the current action and state in the respective lists for future use
        self.prevAction.append(move)
        self.prevState.append(state)

        # Update the current score in qScore
        self.qScore = state.getScore()

        # Return the selected action
        return move


    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended with score {state.getScore()}!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()

        # Extract features from state
        stateFeatures = GameStateFeatures(state)
        # Update the state feature count
        self.updateCount(stateFeatures, None)
        
        # Calculate the reward of current state
        stateReward = state.getScore() - self.qScore
        if len(self.prevState) > 0:
            # Get the previous state and action
            prevState, prevAction = self.prevState[-1], self.prevAction[-1]
            # Update the Q-values based on the previous state, action and stateReward
            self.qUpdate(prevState, prevAction, stateReward, 0)

        # Reset the qScore and previous states and actions
        self.qScore = 0
        self.prevState.clear()
        self.prevAction.clear()

        # Decrease the exploration rate epsilon linearly over the training episodes
        totalEps = self.getEpisodesSoFar()
        numTraining = self.getNumTraining()
        if totalEps > 0 and totalEps <= numTraining:
            episode = 1 - totalEps / numTraining
            self.setEpsilon(episode * 0.1)

        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
