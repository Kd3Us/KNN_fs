import math
import random

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

	def evaluate(self, X_test, Y_test):
		predictions = self.predict(X_test)
		if hasattr(Y_test, "tolist"):
			Y_test_list = Y_test.tolist()
		else:
			Y_test_list = list(Y_test)
		score = self._f1_weighted(Y_test_list, predictions)
		print(f"F1 weighted : {score}")
		return score

	def _euclidean_distance(self, point_a, point_b):
		squared_sum = 0
		for i in range(len(point_a)):
			squared_sum += (point_a[i] - point_b[i]) ** 2
		return math.sqrt(squared_sum)

	def _precision(self, Y_true, Y_pred, target_class):
		true_positive = 0
		false_positive = 0
		for i in range(len(Y_true)):
			if Y_pred[i] == target_class and Y_true[i] == target_class:
				true_positive += 1
			elif Y_pred[i] == target_class and Y_true[i] != target_class:
				false_positive += 1
		if true_positive + false_positive == 0:
			return 0
		return true_positive / (true_positive + false_positive)

	def _recall(self, Y_true, Y_pred, target_class):
		true_positive = 0
		false_negative = 0
		for i in range(len(Y_true)):
			if Y_true[i] == target_class and Y_pred[i] == target_class:
				true_positive += 1
			elif Y_true[i] == target_class and Y_pred[i] != target_class:
				false_negative += 1
		if true_positive + false_negative == 0:
			return 0
		return true_positive / (true_positive + false_negative)

	def _f1_weighted(self, Y_true, Y_pred):
		classes = set(Y_true)
		total = len(Y_true)

		weighted_f1 = 0
		for c in classes:
			precision = self._precision(Y_true, Y_pred, c)
			recall = self._recall(Y_true, Y_pred, c)
			if precision + recall == 0:
				f1_class = 0
			else:
				f1_class = 2 * precision * recall / (precision + recall)

			support = 0
			for label in Y_true:
				if label == c:
					support += 1

			weighted_f1 += (support / total) * f1_class
		return weighted_f1

	def _stratified_k_fold_split(self, X, Y, n_splits=10, random_state=42):
		if hasattr(Y, "tolist"):
			Y_list = Y.tolist()
		else:
			Y_list = list(Y)

		indices_by_class = {}
		for i in range(len(Y_list)):
			label = Y_list[i]
			if label not in indices_by_class:
				indices_by_class[label] = []
			indices_by_class[label].append(i)

		shuffler = random.Random(random_state)
		for label in indices_by_class:
			shuffler.shuffle(indices_by_class[label])

		validation_indices_per_fold = []
		for fold_index in range(n_splits):
			validation_indices_per_fold.append([])

		for label in indices_by_class:
			class_indices = indices_by_class[label]
			for position in range(len(class_indices)):
				fold_index = position % n_splits
				index = class_indices[position]
				validation_indices_per_fold[fold_index].append(index)

		folds = []
		for fold_index in range(n_splits):
			validation_indices = validation_indices_per_fold[fold_index]
			validation_set = set(validation_indices)

			train_indices = []
			for i in range(len(X)):
				if i not in validation_set:
					train_indices.append(i)

			X_train_fold = []
			Y_train_fold = []
			for i in train_indices:
				X_train_fold.append(X[i])
				Y_train_fold.append(Y_list[i])

			X_validation_fold = []
			Y_validation_fold = []
			for i in validation_indices:
				X_validation_fold.append(X[i])
				Y_validation_fold.append(Y_list[i])

			folds.append((X_train_fold, Y_train_fold, X_validation_fold, Y_validation_fold))
		return folds


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	knn_object = KNN(n_neighbors=5)
	knn_object.fit(X_normalized, Y)
	score = knn_object.evaluate(X_normalized[:100], Y.iloc[:100])