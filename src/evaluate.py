import pandas as pd
import torch
from train import SimpleModel  
from yaml import safe_load
from sklearn.metrics import accuracy_score

with open('params.yaml') as f:
    params = safe_load(f)

model = SimpleModel()
model.load_state_dict(torch.load(params['model']['path']))
model.eval()

data = pd.read_csv(params['data']['path'])
X_test = data.iloc[:, :-1]
y_test = data.iloc[:, -1]

with torch.no_grad():
    outputs = model(torch.tensor(X_test.values, dtype=torch.float)).squeeze().round()
    accuracy = accuracy_score(y_test, outputs.numpy())

with open('metrics/accuracy.txt', 'w') as f:
    f.write(f'Accuracy: {accuracy}\n')
