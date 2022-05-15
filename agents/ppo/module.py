from git import Actor
import numpy as np

import torch
import torch.nn as nn
from agents.utils.act import ACTLayer
from torch.distributions import MultivariateNormal, Categorical

class ActorCritic(nn.Module):

    def __init__(self,
                 obs_shape,
                 states_shape,
                 action_space,
                 config,
                 asymmetric=False):

        super(ActorCritic, self).__init__()

        self.asymmetric = asymmetric
        if config is None:
            actor_hidden_dim = [256] * 3
            critic_hidden_dim = [256] * 3
            activation = get_activation("selu")
        else:
            actor_hidden_dim = config["pi_hid_sizes"]
            critic_hidden_dim = config["vf_hid_sizes"]
            activation = get_activation(config['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)-1):
            actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l+1]))
            actor_layers.append(activation)
        
        self.actor_base = nn.Sequential(*actor_layers)
        self.actor_head = ACTLayer(action_space, actor_hidden_dim[-1], config["use_orthogonal"])

        # Value function
        critic_layers = []
        if self.asymmetric:
            critic_layers.append(nn.Linear(*states_shape, critic_hidden_dim[0]))
        else:
            critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l+1]))
                critic_layers.append(activation)

        self.critic = nn.Sequential(*critic_layers)

        # Initialize the weights like in stable baselines
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for module, idx
        in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
    
    def forward(self):
        raise NotImplementedError

    def act(self, observations, states=None, available_actions=None, deterministic=False):
        actor_features = self.actor_base(observations)
        actions, action_log_probs = self.actor_head(actor_features, available_actions, deterministic)

        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)
        
        return actions, action_log_probs, value
    
    def evaluate(self, observations, states, actions):
        actor_features = self.actor_base(observations)
        actions_log_prob, dist_entropy = self.actor_head.evaluate_actions(actor_features, actions)
        
        if self.asymmetric:
            value = self.critic(states)
        else:
            value = self.critic(observations)
        
        return actions_log_prob, dist_entropy, value
    


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
