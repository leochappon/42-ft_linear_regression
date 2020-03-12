import sys
import json

def get_car_mileage(feature_min, feature_max):
	car_mileage = input("What is your car mileage ? ")

	try:
		car_mileage = float(car_mileage)
	except:
		sys.exit("Your car mileage is not valid")

	car_mileage = (car_mileage - feature_min) / (feature_max - feature_min)

	return car_mileage

def get_cache_values():
	try:
		with open("cache.json", 'r') as json_file:
			cache = json.load(json_file)
	except:
		cache = {"theta0": 0, "theta1": 0, "feature_min": 0, "feature_max": 1}

	try:
		return cache["theta0"], cache["theta1"], cache["feature_min"], cache["feature_max"]
	except:
		sys.exit("Invalid cache")

def estimate_price(theta0, theta1, car_mileage):
	return theta0 * car_mileage + theta1

def main():
	theta0, theta1, feature_min, feature_max = get_cache_values()

	car_mileage = get_car_mileage(feature_min, feature_max)

	estimated_price = estimate_price(theta0, theta1, car_mileage)

	print("Your car is estimated at {}".format(int(estimated_price)))

if __name__ == "__main__":
	main()