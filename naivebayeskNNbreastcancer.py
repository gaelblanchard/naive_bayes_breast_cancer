import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def print_results(predictions,data,desired_variable):
	print("Results:")
	print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
	.format(
		test_data.shape[0],
		(data[desired_variable] != predictions).sum(),
		100*(1-(data[desired_variable] != predictions).sum()/data.shape[0])
		)
	)

def plot_correlations(data,correl_columns):
	correlations = pd.DataFrame()
	for main_variable in correl_columns:
		for other_variable in correl_columns:
			correlations.loc[main_variable, other_variable] = data.corr().loc[main_variable, other_variable]
	sb.heatmap(correlations)
	plt.show()
	return correlations

def normalize_columns(data, normalize_columns):
	for column in normalize_columns:
		sum_of_column = data[column].sum()
		data[column] = data[column]/sum_of_column
	return data

np.random.seed(0)
cancer_data = pd.read_csv("bc_data.csv")
cancer_data['numerical_diagnosis'] = np.where(cancer_data['diagnosis']=='M', 1, 0)
cancer_for_norm_data = cancer_data
correl_columns = ['numerical_diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean','radius_se','texture_se','perimeter_se','area_se','smoothness_se','compactness_se','symmetry_se','fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst','compactness_worst','concavity_worst','concave_points_worst','symmetry_worst','fractal_dimension_worst']
cancer_data_correl = cancer_data[correl_columns]
correlations = plot_correlations(cancer_data_correl,correl_columns)
columns_to_normalize = ["radius_worst","concave_points_worst","area_worst","perimeter_worst"]
cancer_for_norm_data = normalize_columns(cancer_for_norm_data, columns_to_normalize)

#Naive Bayes
#subset to change our cancer_data to train our statistical model
factors_list = ["numerical_diagnosis","radius_worst","perimeter_worst","area_worst","concave_points_worst"]
predictor_factors = ["radius_worst","perimeter_worst","area_worst","concave_points_worst"]
breast_cancer_factors = cancer_data[factors_list]
breast_cancer_complete = breast_cancer_factors.dropna()
data_sample = breast_cancer_complete.sample(frac=0.7, replace=False)
training_data = data_sample
test_data = breast_cancer_complete[~breast_cancer_complete.isin(training_data)].dropna()

nb_model = GaussianNB()
nb_model.fit(training_data[predictor_factors],training_data["numerical_diagnosis"])

test_pred = nb_model.predict(test_data[predictor_factors])
print_results(test_pred,test_data,"numerical_diagnosis")
#93.57 % test
#94.33 % train


#kNN
#subset to change our cancer_data to train our statistical model
#id diagnosis radius worst perimeter worst area worst concave points worst
#Used 10 k nearest neighbors for classification 
breast_cancer_norm = cancer_for_norm_data[factors_list]
data_sample = breast_cancer_norm.sample(frac=0.7, replace=False)
training_data = data_sample
test_data = breast_cancer_norm[~breast_cancer_norm.isin(training_data)].dropna()
knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(training_data[predictor_factors],training_data["numerical_diagnosis"])
test_pred = knn_model.predict(test_data[predictor_factors])
print_results(test_pred,test_data,"numerical_diagnosis")

#95.32 test data k = 10 95.91 k = 15
