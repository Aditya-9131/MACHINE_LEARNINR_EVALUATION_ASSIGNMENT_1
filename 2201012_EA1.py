import numpy as np

np.random.seed(42)

n_samples = 100
size = np.random.randint(800, 3500, size=n_samples)
bedrooms = np.random.randint(1, 6, size=n_samples)   
age = np.random.randint(1, 50, size=n_samples)       
distance = np.random.uniform(0.5, 20, size=n_samples) 

price = 50 * size + 10000 * bedrooms - 200 * age - 1500 * distance + np.random.randn(n_samples) * 10000

X = np.column_stack((size, bedrooms, age, distance))
y = price

train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

X_train_bias = np.column_stack((np.ones(train_size), X_train))

theta = np.linalg.pinv(X_train_bias.T.dot(X_train_bias)).dot(X_train_bias.T).dot(y_train)

print("Model Parameters (theta):", theta)

X_test_bias = np.column_stack((np.ones(X_test.shape[0]), X_test))  
y_pred = X_test_bias.dot(theta)  

mse = np.mean((y_test - y_pred) ** 2)
print("Mean Squared Error (MSE):", mse)

print("\nIntercept (bias term):", theta[0])
print("Coefficient for Size:", theta[1])
print("Coefficient for Number of Bedrooms:", theta[2])
print("Coefficient for Age of House:", theta[3])
print("Coefficient for Distance to City Center:", theta[4])

new_house = np.array([1, 2500, 4, 10, 5])  
predicted_price = new_house.dot(theta)
print("\nPredicted price for the new house (2500 sq. ft., 4 bedrooms, 10 years old, 5 miles from city center):", predicted_price)
