
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import model_selection
import time  

def kNN(X, y):

	results = []
	times = []
	accuracy_diff = []

	for x in range(0,200):

		# Dividing the dataset into Training Set and Test Set
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = int(time.time()))

		start = time.time()

		# Applying kNN 
		knn = KNeighborsClassifier(n_neighbors = 5) 
		knn.fit(X_train, y_train)


		# Predicting the results
		y_pred = knn.predict(X_test)


		# Checking the accuracy on the test set
		result = knn.score(X_test, y_test)		


		# Checking the integrity of the model using 10-fold cross validation
		kfold = model_selection.KFold(n_splits = 10, random_state = int(time.time()))
		modelCV = KNeighborsClassifier(n_neighbors = 5)
		scoring = 'accuracy'
		result_cv = model_selection.cross_val_score(modelCV, X_train, y_train, cv = kfold, scoring = scoring)
		result_cv = np.mean(result_cv)


		# Calculating the difference between model prediction and CV result
		aux_accuracy_diff = result - result_cv
		if aux_accuracy_diff < 0:
			aux_accuracy_diff = -aux_accuracy_diff


		# Calculating elapsed time
		end = time.time()
		elapsed_time = end - start


		results.append(result.mean())
		times.append(elapsed_time)
		accuracy_diff.append(aux_accuracy_diff)
		

	# Returning the average performance of the Algorithm
	return[np.mean(results), np.var(results), np.mean(times), np.mean(accuracy_diff)]



