from sklearn.datasets import make_blobs
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np

X1, y1 = make_blobs(n_samples= 100, 
                    n_features= 2,
                    centers=1, 
                    center_box=(2.0, 2.0),
                    cluster_std= 0.75,
                    random_state=69)

X2, y2 = make_blobs(n_samples= 100, 
                    n_features= 2,
                    centers=1, 
                    center_box=(3.0, 3.0),
                    cluster_std= 0.75,
                    random_state=69)

# Combine datasets A and B
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(100), np.ones(100)))  # Label A as 0 and B as 1

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import matplotlib.pyplot as plt

# Plot the dataset
fig = plt.figure()
fig.suptitle("Data Sample")
plt.scatter(X1[:, 0], X1[:, 1], c = 'red', linewidths = 1, alpha = 0.6, label = "Class 1")
plt.scatter(X2[:, 0], X2[:, 1], c = 'blue', linewidths = 1, alpha = 0.6, label = "Class 2")
plt.xlabel('Feature 1', fontsize=10)
plt.ylabel('Feature 2', fontsize=10)
plt.grid(True, axis='both')
plt.legend(loc='lower right')
plt.show()
plt.savefig('Out1 - Data Sample.png')


# Creating a neural network model
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))
optimizer = Adam(0.0001)
model.compile(loss='binary_crossentropy', optimizer = 'adam',
metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1, validation_split=0.2)

# Plot the decision boundary
# Create a grid of points to evaluate
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), 
                     np.arange(y_min, y_max, 0.01))

# Predict on the grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['red', 'blue'], alpha=0.3)

# Plot the scatter points for each class explicitly
plt.scatter(X1[:, 0], X1[:, 1], c='red', edgecolor='k', label='Class 1')
plt.scatter(X2[:, 0], X2[:, 1], c='blue', edgecolor='k', label='Class 2')

plt.title('Decision Boundary of Neural Network')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()  # Automatically use the labels defined in scatter
plt.grid()
plt.show()