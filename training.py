import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from a pickle file
with open('./data.pickle', 'rb') as file:
    data_dict = pickle.load(file)

# Convert lists to NumPy arrays for compatibility with scikit-learn
# Ensuring that all elements of data are of consistent shape is crucial
data = np.array(data_dict['data'], dtype=object)
labels = np.array(data_dict['labels'])

# Check if data shapes are consistent
if np.any([len(i) != len(data[0]) for i in data]):
    raise ValueError("All data points must have the same number of features")

# Splitting the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
)

# Create a RandomForestClassifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(x_train.tolist(), y_train)

# Predict labels for the test set
y_predict = model.predict(x_test.tolist())

# Calculate and print the accuracy of the model
accuracy = accuracy_score(y_predict, y_test) * 100
print(f'{accuracy}% of samples were classified correctly')


with open('model.pkl', 'wb') as f:
    pickle.dump({'model': model}, f)
    print("Model has been saved to 'model.pkl'.")