from torch import nn

class DQN(nn.Module):
    def __init__(self, n_observations:int, n_actions:int, hidden:int = 128) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_observations, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, hidden),
                                    nn.ReLU(),
                                    nn.Linear(hidden, n_actions))
        
    def forward(self, x):
        return self.layers(x)
    