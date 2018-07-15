
from Logistic_Regression import logistic_Regression
from Knn import kNN
from Naive_Bayes import naive_Bayes

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Loading the Bank Dataset
dataset = pd.read_csv('BankDataset/dataset.csv')
classes = pd.read_csv('BankDataset/classes.csv')

# Loading the Breast Cancer Datast
#dataset = pd.read_csv('BreastCancerDataset/dataset.csv')
#classes = pd.read_csv('BreastCancerDataset/classes.csv')

X = dataset.iloc[:, :].values
y = classes.iloc[:, 0].values


# Performing Logistic Regression
[result_LR, var_LR, time_LR, accuracy_diff_LR] = logistic_Regression(X, y)

print()
print("Logistic Regression Results:")
print("Accuracy of the Classifier: " + str(result_LR))
print("Variance: " + str(var_LR))
print("Time required for a single run: " + str(time_LR))
print("Average difference between model prediction and CV result: " + str(accuracy_diff_LR))


# Performing kNN
[result_kNN, var_kNN, time_kNN, accuracy_diff_kNN] = kNN(X, y)

print()
print("Knn Results:")
print("Accuracy of the Classifier: " + str(result_kNN))
print("Variance: " + str(var_kNN))
print("Time required for a single run: " + str(time_kNN))
print("Average difference between model prediction and CV result: " + str(accuracy_diff_kNN))


# Performing Naive Bayes
[result_NB, var_NB, time_NB, accuracy_diff_NB] = naive_Bayes(X, y)

print()
print("Naive Bayes Results:")
print("Accuracy of the Classifier: " + str(result_NB))
print("Variance: " + str(var_NB))
print("Time required for a single run: " + str(time_NB))
print("Average difference between model prediction and CV result: " + str(accuracy_diff_NB))


# Plotting results using Barcharts

objects = ('Logistic Regression', 'Naive Bayes', 'kNN')
y_pos = np.arange(len(objects))

performance_perc = [result_LR, result_NB, result_kNN]
variance = [var_LR, var_NB, var_kNN]
performance_time = [time_LR, time_NB, time_kNN]
accuracy_diff = [accuracy_diff_LR, accuracy_diff_NB, accuracy_diff_kNN]

plt.figure(0)
plt.ylim(ymax = 1)
plt.bar(y_pos, performance_perc, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance (%)')
plt.title('Accuracy of the Classifier') 
plt.show()

plt.figure(1)
plt.bar(y_pos, variance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Variance')
plt.title('Variance') 
plt.show()

plt.figure(2)
plt.bar(y_pos, performance_time, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Performance (s)')
plt.title('Time required for a single run') 
plt.show()

plt.figure(3)
plt.bar(y_pos, accuracy_diff, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Difference')
plt.title('Average difference between model prediction and CV result:') 
plt.show()


