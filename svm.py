import numpy as np
from sklearn import svm
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def read_data(names, strokes):
	data = []
	for name in names:
		for stroke in range(len(strokes)):
			filename = f'results/{name}/{strokes[stroke]}/arm_angles.txt'
			with open(filename, 'r') as file:
				lines = file.readlines()
			counter = 0
			zero_counter = 0
			cycle_sum_left = 0
			cycle_sum_right = 0
			for line in lines:
				left, right = line.strip().split(',')
				left = float(left)
				right = float(right)
				if left == 0.0 and right == 0.0:
					if counter != 0 and zero_counter == 3:
						data.append([cycle_sum_left/counter, cycle_sum_right/counter, stroke])
						counter = 0
						cycle_sum_left = 0
						cycle_sum_right = 0
						zero_counter = 0
					else:
						zero_counter += 1
				else:
					cycle_sum_left += left
					cycle_sum_right += right
					counter += 1
					zero_counter = 0
	return np.array(data)

def plot_data(data):
	label = data[:,2]
	idx1 = np.where(label == 0)[0]
	idx2 = np.where(label == 1)[0]
	idx3 = np.where(label == 2)[0]
	idx4 = np.where(label == 3)[0]

	plt.scatter(data[:,0][idx1],data[:,1][idx1],c='r')
	plt.scatter(data[:,0][idx2],data[:,1][idx2],c='g')
	plt.scatter(data[:,0][idx3],data[:,1][idx3],c='b')
	plt.scatter(data[:,0][idx4],data[:,1][idx4],c='y')
	plt.show()


def run_svm(data):
	X = data[:,0:2]
	y = data[:,2]
	X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20)
	
	# Two different classifiers - Polynomial/RBF kernel
	rbf = svm.SVC(kernel='rbf', gamma=0.5, C=0.1).fit(X_train, y_train)
	poly = svm.SVC(kernel='poly', degree=10, C=1).fit(X_train, y_train)

	poly_pred = poly.predict(X_test)
	rbf_pred = rbf.predict(X_test)

	poly_accuracy = accuracy_score(y_test, poly_pred)
	poly_f1 = f1_score(y_test, poly_pred, average='weighted')
	print('Accuracy (Polynomial Kernel): ', "%.2f" % (poly_accuracy*100))
	print('F1 (Polynomial Kernel): ', "%.2f" % (poly_f1*100))

	rbf_accuracy = accuracy_score(y_test, rbf_pred)
	rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
	print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
	print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

def run_script():
	names = ["Daytona","Hannah","Hayden","Jake","Mike","Ruby","Sierra","Vera"]
	# Relate to class 0, 1, 2, 3
	strokes = ["Fly","Backstroke","Breaststroke","Free"]
	names = ["Daytona"]
	data = read_data(names, strokes)
	run_svm(data)
	# plot_data(data)

run_script()