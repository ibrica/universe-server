# Based on https://github.com/ikostrikov/pytorch-a3c
import torch
import torch.nn as nn
import torch.nn.functional as F

class ES(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        """
        Initialize network
        """
        super(ES, self).__init__()
        num_outputs = action_space.n

        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32*3*3, 256)
        self.actor_linear = nn.Linear(256, num_outputs)
        self.train()



    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32*3*3)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.actor_linear(x), (hx, cx)

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        """
        The params that should be trained by ES (all of them)
        """
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]