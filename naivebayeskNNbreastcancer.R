library(e1071)
library(caret)
library(ggpubr)
library(RWeka)
library(ggcorrplot)
setwd("/directory/for/file")
set.seed(0)
cancer_data <- data.frame(read.table("bc_data.csv", header = TRUE, sep = ","))
cancer_for_norm_data <- data.frame(read.table("bc_data.csv", header = TRUE, sep = ","))
names(cancer_data) <- c("id", "diagnosis","radius_mean", "texture_mean", "perimeter_mean","area_mean", "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean", "symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
names(cancer_for_norm_data) <- c("id", "diagnosis","radius_mean", "texture_mean", "perimeter_mean","area_mean", "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean", "symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
cancer_data$numerical_diagnosis <- ifelse((cancer_data$diagnosis=="M"),1,0)
#determine correlation this is essential for wanting to determine
#what columns we want to use as predictors for
#a benign and malignant cancer cell
#Visual representation using a correlogram
#correlogram
subset_cancer_data <- subset(cancer_data, select=c(
				numerical_diagnosis,
				radius_mean,
				texture_mean,
				perimeter_mean,
				area_mean,
				smoothness_mean,
				compactness_mean,
				concavity_mean,
				concave_points_mean,
				symmetry_mean,
				fractal_dimension_mean,
				radius_se,
				texture_se,
				perimeter_se,
				area_se,
				smoothness_se,
				compactness_se,
				symmetry_se,
				fractal_dimension_se,
				radius_worst,
				texture_worst,
				perimeter_worst,
				area_worst,
				smoothness_worst,
				compactness_worst,
				concavity_worst,
				concave_points_worst,
				symmetry_worst,
				fractal_dimension_worst
				)
			)
cor(subset_cancer_data)
c_gram_breast_cancer <- round(cor(subset_cancer_data), 1)
# Plots the correlogram
ggcorrplot(
			c_gram_breast_cancer, 
			hc.order = TRUE, 
            type = "lower", 
            lab = TRUE, 
            lab_size = 3, 
            method="circle", 
            colors = c("tomato2", "white", "springgreen3"), 
            title="Correlogram of Gun Violence Data", 
            ggtheme=theme_bw
            )
#sum of columns for normalization
sum_radius_worst <- sum(cancer_for_norm_data$radius_worst) 
sum_concave_points_worst <- sum(cancer_for_norm_data$concave_points_worst)
#sum_concave_points_worst <- sum(cancer_for_norm_data$radius_mean)
sum_area_worst <- sum(cancer_for_norm_data$area_worst)
sum_perimeter_worst <- sum(cancer_for_norm_data$perimeter_worst)
#normalizing the data
cancer_for_norm_data$radius_worst <- cancer_for_norm_data$radius_worst/sum_radius_worst
cancer_for_norm_data$concave_points_worst <- cancer_for_norm_data$concave_points_worst/sum_concave_points_worst
cancer_for_norm_data$area_worst <- cancer_for_norm_data$area_worst/sum_area_worst
cancer_for_norm_data$perimeter_worst <- cancer_for_norm_data$perimeter_worst/sum_perimeter_worst
#Test if normalizing the data changes results and if as factor changes the results
breast_cancer_norm <- subset(cancer_for_norm_data,select = c("id", "diagnosis","radius_worst", "concave_points_worst", "perimeter_worst","area_worst"))
#Naive Bayes Model 0.7427 post worst correlates 0.7661
cancer_data$diagnosis <- as.factor(cancer_data$diagnosis)
cancer_data$radius_worst <- as.factor(cancer_data$radius_worst)
cancer_data$perimeter_worst <- as.factor(cancer_data$perimeter_worst)
cancer_data$area_worst <- as.factor(cancer_data$area_worst)
cancer_data$concave_points_worst <- as.factor(cancer_data$concave_points_worst)
breast_cancer <- subset(cancer_data,select = c("id", "diagnosis","radius_worst", "concave_points_worst", "perimeter_worst","area_worst"))
breast_cancer_factors <- breast_cancer[, 2:6]
breast_cancer_complete <- breast_cancer_factors[complete.cases(breast_cancer_factors),]
breast_cancer_complete$diagnosis <- as.factor(breast_cancer_complete$diagnosis)
data.samples <- sample(1:nrow(breast_cancer_complete),nrow(breast_cancer_complete) * 0.7, replace = FALSE)
training.data <- breast_cancer_complete[data.samples, ]
test.data <- breast_cancer_complete[-data.samples, ]
nb.model <- naiveBayes(diagnosis ~ ., data = training.data)
prediction.nb <- predict(nb.model, test.data)
table(test.data$diagnosis, prediction.nb)
confusionMatrix(prediction.nb,test.data$diagnosis)

#kNN model 10 folds pre norm 0.732865 post norm 0.908612 
# post worst correlates 0.940246
classifier <- IBk(diagnosis ~., data = breast_cancer_norm, control = Weka_control(K = 20, X = TRUE))
evaluate_Weka_classifier(classifier, numFolds = 10)
table(breast_cancer_norm$diagnosis, classifier$predictions)
confusionMatrix(classifier$predictions, breast_cancer_norm$diagnosis)