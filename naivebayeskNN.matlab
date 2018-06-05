CancerData = readtable('bc_data.csv')
varNames = {'radius_worst','concave_points_worst','perimeter_worst','area_worst'}
Diagnosis = CancerData.diagnosis
PredictorVariables = CancerData(:,varNames)

%Naive Bayes Model
NaiveModelDiagnosis = fitcnb(PredictorVariables,Diagnosis,'ClassNames',{'B','M'})
NaiveModelDiagnosis.DistributionParameters
NaiveModelDiagnosis.DistributionParameters{1,1}
NaiveModelResubErr = resubLoss(NaiveModelDiagnosis)
% 6 percent misclassification 94 percent
PredValue = predict(NaiveModelDiagnosis,CancerData(:,varNames))
NBConfMat = confusionmat(Diagnosis,PredValue)
CVNaiveModel = crossval(NaiveModelDiagnosis)
kNaiveLossModel = kfoldLoss(CVNaiveModel)
% 6 percent

%kNN Model
kNNDiagnosis = fitcknn(PredictorVariables,Diagnosis,'NSMethod','exhaustive','Distance','cosine','NumNeighbors',2)
kNNResubError = resubLoss(kNNDiagnosis)
% 6 percent misclassification 92 percent interstingly no false negatives
PredValue = predict(kNNDiagnosis,CancerData(:,varNames))
kNNConfMat = confusionmat(Diagnosis,PredValue)
CVkNNModel = crossval(kNNDiagnosis)
kNNLossModel = kfoldLoss(CVkNNModel)
% 10.5 percent 