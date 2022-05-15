from datetime import datetime
import os
import time
from git import Actor

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from agents.ppo import RolloutStorage, ActorCritic

class PPO():

    def __init__(self,
                 vec_env,
                 actor_critic_class:ActorCritic,
                 config,
                 log_dir,
                 device=torch.device("cpu")):
        
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        self.device = device
        self.asymmetric = config["asymmetric"]

        # PPO
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(self.observation_space.shape,
                                               self.state_space.shape,
                                               self.action_space,
                                               config,
                                               self.asymmetric
                                               )

        self.actor_critic.to(self.device)
        self.storage = RolloutStorage()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config["learning_rate"])

        # PPO parameters
        self.clip_param = config["clip_param"]
        self.num_learning_epochs = config["num_learning_epochs"]
        self.num_mini_batches = config["num_mini_batches"]
        self.num_transitions_per_env = config["num_transitions_per_env"]
        self.value_loss_coef = config["value_loss_coef"]
        self.entropy_coef = config["entropy_coef"]
        self.gamma = config["gamma"]
        self.lam = config["lam"]
        self.max_grad_norm = config["max_grad_norm"]
        self._use_clipped_value_loss = config["use_clipped_value_loss"]

        # Log
        self.log_dir = log_dir
        self.print_log = config["print_log"]
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
