from sklearn.datasets import make_classification
import pandas as pd

n_samples = 1000
n_features = 10
n_informative = 8
n_redundant = 2
n_classes = 2
weights = [0.7, 0.3]
random_state = 42

X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                           n_informative=n_informative, n_redundant=n_redundant, 
                           n_classes=n_classes, weights=weights, 
                           random_state=random_state)



feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(df.head())

df.to_csv('data/dataset.csv', index=False)
