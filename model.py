import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Define the layers
        hidden = [128, 256, 128]

        self.fc_in = nn.Linear(state_size, hidden[0])
        self.hidden_list = nn.ModuleList([nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden)-1)])
        self.fc_out = nn.Linear(hidden[-1], action_size)

    def forward(self, state):
        """A network that maps state -> action values."""

        x = F.relu(self.fc_in(state))

        # Go through the hidden layers
        for layer in self.hidden_list:
            x = F.relu(layer(x))

        return self.fc_out(x)


class DuellingQNetwork(nn.Module):
    """Actor (Policy) Model using the duelling Network architecture."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()

        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        hidden = [128, 256, 128]

        # Define the advantage-function stream
        self.fc_in_adv = nn.Linear(state_size, hidden[0])
        self.hidden_list_adv = nn.ModuleList([nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden)-1)])
        self.fc_out_adv = nn.Linear(hidden[-1], action_size)

        # Define the state-value-function stream
        self.fc_in_val = nn.Linear(state_size, hidden[0])
        self.hidden_list_val = nn.ModuleList([nn.Linear(hidden[i], hidden[i+1]) for i in range(len(hidden)-1)])
        self.fc_out_val = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        """A network that maps state -> action values."""

        # Calculate the advantage-stream
        adv = F.relu(self.fc_in_adv(state))
        for layer in self.hidden_list_adv:
            adv = F.relu(layer(adv))
        adv = self.fc_out_adv(adv)

        # Calculate the state-value-stream
        val = F.relu(self.fc_in_val(state))
        for layer in self.hidden_list_val:
            val = F.relu(layer(val))
        val = self.fc_out_val(val).expand(state.size(0), self.action_size)

        # Combining the to streams to compute the output
        out = val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)

        return out