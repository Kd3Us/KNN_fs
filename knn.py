import math

from data_loader import load_normalized_data


class KNN:

	def __init__(self, n_neighbors=5):
		self.n_neighbors = n_neighbors
		self.X_train = None
		self.Y_train = None

	def fit(self, X, Y):
		self.X_train = X
		self.Y_train = Y

	def _euclidean_distance(self, point_a, point_b):
		squared_sum = 0
		for i in range(len(point_a)):
			squared_sum += (point_a[i] - point_b[i]) ** 2
		return math.sqrt(squared_sum)


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=5)
	knn_object.fit(X_normalized, Y)
	distance = knn_object._euclidean_distance(X_normalized[0], X_normalized[1])
	print(f"Distance between point 0 and point 1 : {distance}")