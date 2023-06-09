#!/usr/bin/env python

"""An abstract class that specifies the Agent API for RL-Glue-py.
"""

from __future__ import print_function
from abc import ABCMeta, abstractmethod


class BaseAgent:
    """
    Abstract class for Flappy Agent.
    Note:
        agent_init, agent_start, agent_step, agent_end are required methods.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, agent_info={}):
        pass

    @abstractmethod
    def policy(observation):
        pass

    @abstractmethod
    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            observation (Numpy array): the state observation from the environment's evn_start function.
        Returns:
            The first action the agent takes.
        """

    @abstractmethod
    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            observation (Numpy array): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """

    @abstractmethod
    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the terminal state.
        """

    @abstractmethod
    def agent_cleanup(self):
        """Cleanup done after the agent ends."""
