from data_loader import load_normalized_data


class KNN:

	def __init__(self, n_neighbors=5):
		self.n_neighbors = n_neighbors
		self.X_train = None
		self.Y_train = None

	def fit(self, X, Y):
		self.X_train = X
		self.Y_train = Y


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=5)
	knn_object.fit(X_normalized, Y)
	print("Fit done")