import torch
from torch import nn
class Net(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        self. fc1 = nn.Linear(input_size, 10)
        nn.init.normal_(self.fc1.weight, 0, 2)
        self.lstm = nn.LSTMCell(10, 50, )
        self.fc2 = nn.Linear(10, 20)
        self.fc3 = nn.Linear(20, 40)
        self.fc4 = nn.Linear(40, output_size)
        nn.init.normal_(self.fc4.weight, 0, 2)


    def forward(self, X):
        X = self.fc1(X)
        # X = self.lstm(X)
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.fc4(X)
        return X

    def train_loop(self, X, Y, loss_fn, optimizer):

        # Compute prediction and loss
        pred = self(torch.tensor(X).float())
        loss = loss_fn(pred, torch.tensor(Y).float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        return loss