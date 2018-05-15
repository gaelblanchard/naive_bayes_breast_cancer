library(e1071)
library(caret)
library(ggpubr)
setwd("/directory/to/data/file")
cancer_data <- data.frame(read.table("bc_data.csv", header = TRUE, sep = ","))

names(cancer_data) <- c("id", "diagnosis","radius_mean", "texture_mean", "perimeter_mean","area_mean", "smoothness_mean", "compactness_mean","concavity_mean", "concave_points_mean", "symmetry_mean","fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
#created a numerical diagnosis column variable to test correlation
cancer_data$numerical_diagnosis <- ifelse((cancer_data$diagnosis=="M"),1,0)
cancer_data$experimental_perimeter_radius <- cancer_data$perimeter_mean * cancer_data$radius_mean
#cancer_data$experimental_area_compactness <- cancer_data$area_mean * cancer_data$compactness_mean
#determine correlation
cor(cancer_data$numerical_diagnosis,cancer_data$radius_mean,method = "pearson")
cor(cancer_data$numerical_diagnosis,cancer_data$texture_mean,method = "pearson")
cor(cancer_data$numerical_diagnosis,cancer_data$perimeter_mean,method = "pearson")
cor(cancer_data$numerical_diagnosis,cancer_data$area_mean,method = "pearson")
cor(cancer_data$numerical_diagnosis,cancer_data$compactness_mean,method = "pearson")
cor(cancer_data$numerical_diagnosis,cancer_data$fractal_dimension_mean,method = "pearson")

cancer_data$diagnosis <- as.factor(cancer_data$diagnosis)
cancer_data$radius_mean <- as.factor(cancer_data$radius_mean)
#cancer_data$texture_mean <- as.factor(cancer_data$texture_mean)
#cancer_data$experimental_area_compactness <- as.factor(cancer_data$experimental_area_compactness)
cancer_data$experimental_perimeter_radius <- as.factor(cancer_data$experimental_perimeter_radius)
cancer_data$perimeter_mean <- as.factor(cancer_data$perimeter_mean)
cancer_data$area_mean <- as.factor(cancer_data$area_mean)
#cancer_data$smoothness_mean <- as.factor(cancer_data$smoothness_mean)
#cancer_data$compactness_mean <- as.factor(cancer_data$compactness_mean)
#cancer_data$concavity_mean <- as.factor(cancer_data$concavity_mean)
#cancer_data$concave_points_mean <- as.factor(cancer_data$concave_points_mean)
#cancer_data$symmetry_mean <- as.factor(cancer_data$symmetry_mean)
#cancer_data$fractal_dimension_mean <- as.factor(cancer_data$fractal_dimension_mean)
#breast_cancer <- data.frame(sapply(cancer_data, as.factor))
# removed concavity and concave points from this subset
# also smoothness and symmetry mean
breast_cancer <- subset(cancer_data,select = c("id", "diagnosis","radius_mean", "experimental_perimeter_radius", "perimeter_mean","area_mean"))
#we can ignore id bc it has no relevance to our predictive mdoel
breast_cancer_factors <- breast_cancer[, 2:6]

nb.model <- naiveBayes(diagnosis ~ ., data = breast_cancer_factors)


breast_cancer_complete <- breast_cancer_factors[complete.cases(breast_cancer_factors),]
breast_cancer_complete$diagnosis <- as.factor(breast_cancer_complete$diagnosis)
data.samples <- sample(1:nrow(breast_cancer_complete),nrow(breast_cancer_complete) * 0.7, replace = FALSE)

training.data <- breast_cancer_complete[data.samples, ]
test.data <- breast_cancer_complete[-data.samples, ]

nb.model <- naiveBayes(diagnosis ~ ., data = training.data)

prediction.nb <- predict(nb.model, test.data)

table(test.data$diagnosis, prediction.nb)

confusionMatrix(prediction.nb,test.data$diagnosis)
#When all csv columns are considered as factors accuracy was 0.6316
#increases to 0.6433 when we lower the amount of factors in breast_cancer_factors
#When we use a subset instead of the data frame to lower the factors that we explicitly state
#we increased the accuracy to 0.6608
#After removing concavity mean and concave points mean increaced accuracy to 0.6725
#After removing symmetry mean and smoothness mean up to 0.6901
#Replacing text_mean and compactness_mean and fractal_dimenasion with experimental measures
# increases to 0.6959
#removing experimental measure exp area_comp increased accuracy to 0.7251