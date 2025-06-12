import os
from sklearn.datasets import load_iris
import pandas as pd

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.to_csv('data/iris.csv', index=False)
print('Sample data saved to data/iris.csv')
