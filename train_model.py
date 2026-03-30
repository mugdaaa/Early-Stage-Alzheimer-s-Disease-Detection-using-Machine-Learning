import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
data = {
    'Age': [65, 70, 75, 80, 60],
    'Memory': [3, 2, 2, 1, 3],
    'Thinking': [3, 2, 2, 1, 3],
    'Decision': [3, 2, 2, 1, 3],
    'Result': [0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Split the dataset into features and target variable
X = df[['Age', 'Memory', 'Thinking', 'Decision']]
y = df['Result']

# Train the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model to a file
with open('dementia_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("Model trained and saved successfully.")
