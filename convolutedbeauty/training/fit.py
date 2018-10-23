import torch.optim as optim
import torch.nn as nn


class Train:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def fit(self, epoch, minibatch, lr=0.0001):
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

        running_loss = 0.0

        for epoch_i in range(1, epoch + 1):
            for i in range(0, len(self.dataset), minibatch):
                optimizer.zero_grad()

                labels = self.dataset[i:minibatch + i]
                output = self.model.forward()

                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
