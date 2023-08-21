import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import datetime

class Linear_QNET(nn.Module):
    def __init__(self, inputSize, hiddenSize1, hiddenSize2, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize1)
        self.linear2 = nn.Linear(hiddenSize1, hiddenSize2)
        self.linear3 = nn.Linear(hiddenSize2, outputSize)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def save(self, fileName=None):
        modelFolderPath = '.\models'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)
        
        now = datetime.datetime.now()
        if fileName == None: fileName = f'model_{now.strftime("%m:%d:%H:%M:%S").replace(":", ".")}.pth'
        fileName = os.path.join(modelFolderPath, fileName)
        torch.save(self.state_dict(), fileName)

    def loadModel(self, fileName=None):
        modelFolderPath = '.\models'
        if fileName == None: fileName = sorted(os.listdir(modelFolderPath))[-1]
        fileName = os.path.join(modelFolderPath, fileName)
        self.load_state_dict(torch.load(fileName))
        print(f"Loaded Model: {fileName}")


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        print("Paramters: ", self.get_n_params(model))
        self.criterion = nn.MSELoss()
    
    # https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    def get_n_params(self, model):
        pp=0
        for p in list(model.parameters()):
            nn=1
            for s in list(p.size()):
                nn = nn*s
            pp += nn
        return pp

    def trainStep(self, state, action, reward, nextState, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        nextState = torch.tensor(np.array(nextState), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            
        # 1: Predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone()
        for x in range(len(done)):
            Qnew = reward[x]
            if not done[x]:
                Qnew = reward[x] + self.gamma * torch.max(self.model(nextState[x]))
                
            target[x][torch.argmax(action[x]).item()] = Qnew
            
        # 2: Qnew = r + y * max(nextPredicted Q value) -> only do this if not done
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
        