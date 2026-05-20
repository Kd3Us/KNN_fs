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

	def predict(self, X):
		predictions = []
		for point in X:
			distances = []
			for i in range(len(self.X_train)):
				distance = self._euclidean_distance(point, self.X_train[i])
				label = self.Y_train.iloc[i]
				distances.append((distance, label))
			distances.sort(key=lambda pair: pair[0])

			nearest_labels = []
			for i in range(self.n_neighbors):
				nearest_labels.append(distances[i][1])

			predicted_label = max(set(nearest_labels), key=nearest_labels.count)
			predictions.append(predicted_label)
		return predictions

	def _euclidean_distance(self, point_a, point_b):
		squared_sum = 0
		for i in range(len(point_a)):
			squared_sum += (point_a[i] - point_b[i]) ** 2
		return math.sqrt(squared_sum)

    

if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=5)
	knn_object.fit(X_normalized, Y)
	predictions = knn_object.predict(X_normalized[:10])
	print(f"Predictions : {predictions}")
	print(f"True labels : {Y.iloc[:10].tolist()}")