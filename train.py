import sys
import numpy as np
import matplotlib.pyplot as plt
import json

learning_rate = 0.7
n_iterations = 1000

def dataset_values(dataset_file):
	try:
		data = np.genfromtxt(dataset_file, dtype='float', delimiter=',', skip_header=1)
	except:
		sys.exit("An error occured while extracting the dataset")

	feature = data[:,0]
	target = data[:,1]

	feature = feature.reshape(-1, 1)
	target = target.reshape(-1, 1)

	return feature, target

def normalize(feature):
	feature_n = np.zeros(feature.shape)

	for i in range(len(feature)):
		feature_n[i] = (feature[i] - min(feature)) / (max(feature) - min(feature))

	return feature_n

def model(X, theta):
	return X.dot(theta)

def cost_function(X, theta, target):
	return np.sum((model(X, theta) - target) ** 2) / (2 * len(target))

def gradient(X, theta, target):
	return X.T.dot(model(X, theta) - target) / len(target)

def gradient_descent(X, target, learning_rate, n_iterations):
	theta = np.random.randn(2, 1)
	cost_history = np.zeros(n_iterations)

	for i in range(n_iterations):
		theta = theta - learning_rate * gradient(X, theta, target)
		cost_history[i] = cost_function(X, theta, target)

	return theta, cost_history

def save(theta, feature):
	cache = {
		"theta0": float(theta[0]),
		"theta1": float(theta[1]),
		"feature_min": float(min(feature)),
		"feature_max": float(max(feature)),
	}

	try:
		with open("cache.json", 'w') as json_file:
			json.dump(cache, json_file, indent=4)
	except:
		sys.exit("Error occured while creating cache")

def coef_determination(target, predictions):
	u = np.sum((target - predictions) ** 2)
	v = np.sum((target - np.mean(target)) ** 2)

	return 1 - u / v

def show(feature, target, predictions, cost_history):
	plt.figure("Linear regression")
	plt.xlabel("km")
	plt.ylabel("price")
	plt.scatter(feature, target)
	plt.plot(feature, predictions, c='r')

	plt.figure("Gradient descent")
	plt.xlabel("n_iterations")
	plt.ylabel("cost_history")
	plt.plot(range(n_iterations), cost_history)

	plt.show()

def main():
	if len(sys.argv) != 2:
		sys.exit("Dataset required")

	feature, target = dataset_values(sys.argv[1])

	feature_n = normalize(feature)

	X = np.hstack((feature_n, np.ones(feature_n.shape)))

	theta, cost_history = gradient_descent(X, target, learning_rate, n_iterations)

	save(theta, feature)

	predictions = model(X, theta)

	print("Coefficient de détermination: {}".format(coef_determination(target, predictions)))

	show(feature, target, predictions, cost_history)

if __name__ == "__main__":
	main()
