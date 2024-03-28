import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from yaml import safe_load
from dvclive import Live

with open('params.yaml') as f:
    params = safe_load(f)

data = pd.read_csv(params['data']['path'])
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1], test_size=0.2)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 10)
        self.fc2 = nn.Linear(10, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

model = SimpleModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=params['train']['lr'])

with Live() as live:
    live.log_param("Epochs", params['train']['epoch'])
    
    for epoch in range(params['train']['epoch']):
        optimizer.zero_grad()
        outputs = model(torch.tensor(X_train.values, dtype=torch.float))
        loss = criterion(outputs, torch.tensor(y_train.values, dtype=torch.float).view(-1, 1))
        loss.backward()
        optimizer.step()
        
        live.log_metric("loss", loss.item())
        live.next_step()

torch.save(model.state_dict(), params['model']['path'])
