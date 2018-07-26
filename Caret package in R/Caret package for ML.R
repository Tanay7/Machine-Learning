setRepositories()  # Setting the default CRAN repo
install.packages('caret')  #Installing the Caret packages

######################Installation and Loading####################
#Loading caret package
library("caret") 

#Loading training data
train<-read.csv("C:\\Users\\Tanay\\Desktop\\train_u6lujuX_CVtuZ9i.csv",stringsAsFactors = T)



############### Pre-processing using Caret###########################

#Checking missing values
sum(is.na(train))  

#Imputing missing values using KNN.Also centering and scaling numerical columns
preProcValues <- preProcess(train, method = c("knnImpute","center","scale"))
library('RANN')
train_processed <- predict(preProcValues, train)
sum(is.na(train_processed))


#Converting outcome variable to numeric
train_processed$Loan_Status<-ifelse(train_processed$Loan_Status=='N',0,1)

id<-train_processed$Loan_ID
train_processed$Loan_ID<-NULL

#Checking the structure of processed train file
str(train_processed)


#Converting every categorical variable to numerical using dummy variables
dmy <- dummyVars(" ~ .", data = train_processed,fullRank = T)
train_transformed <- data.frame(predict(dmy, newdata = train_processed))

#Checking the structure of transformed train file
str(train_transformed)


#Converting the dependent variable back to categorical
train_transformed$Loan_Status<-as.factor(train_transformed$Loan_Status)


###########################Splitting data using caret########################

#Spliting training set into two parts based on outcome: 75% and 25%
index <- createDataPartition(train_transformed$Loan_Status, p=0.75, list=FALSE)
trainSet <- train_transformed[ index,]
testSet <- train_transformed[-index,]

#Checking the structure of trainSet
str(trainSet)



########################Feature selection using Caret##########################

#Feature selection using rfe in caret
control <- rfeControl(functions = rfFuncs,
                      method = "repeatedcv",
                      repeats = 3,
                      verbose = FALSE)
outcomeName<-'Loan_Status'
predictors<-names(trainSet)[!names(trainSet) %in% outcomeName]
Loan_Pred_Profile <- rfe(trainSet[,predictors], trainSet[,outcomeName],
                         rfeControl = control)
Loan_Pred_Profile

#Recursive feature selection
#Outer resampling method: Cross-Validated (10 fold, repeated 3 times)
#Resampling performance over subset size.

#Taking only the top 5 predictors

predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome", "CoapplicantIncome")


######################################Training models using Caret########################
names(getModelInfo())

#Applying a large number of algorithms
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm')
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf')
model_nnet<-train(trainSet[,predictors],trainSet[,outcomeName],method='nnet')
model_glm<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm')


#################################Parameter tuning using Caret###################
fitControl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 5)


###Using tuneGrid to find the parameters of the model
modelLookup(model='gbm')

#Creating grid
grid <- expand.grid(n.trees=c(10,20,50,100,500,1000),shrinkage=c(0.01,0.05,0.1,0.5),n.minobsinnode = c(3,5,10),interaction.depth=c(1,5,10))

# training the model
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneGrid=grid)

# summarizing the model
print(model_gbm)

plot(model_gbm)


#using tune length
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=10)
print(model_gbm)


plot(model_gbm)


#######################################Variable importance estimation using caret############

#Checking variable importance for GBM
#Variable Importance
varImp(object=model_gbm)

#Plotting Variable importance for GBM
plot(varImp(object=model_gbm),main="GBM - Variable Importance")


#Checking variable importance for RF
varImp(object=model_rf)

#Plotting Varianle importance for Random Forest
plot(varImp(object=model_rf),main="RF - Variable Importance")


#Checking variable importance for NNET
varImp(object=model_nnet)

#Plotting Variable importance for Neural Network
plot(varImp(object=model_nnet),main="NNET - Variable Importance")

#Checking variable importance for GLM
varImp(object=model_glm)
#Plotting Variable importance for GLM
plot(varImp(object=model_glm),main="GLM - Variable Importance")



#########################Predictions using Caret#################################

predictions<-predict.train(object=model_gbm,testSet[,predictors],type="raw")
table(predictions)


#confusionMatrix function will give the confusion matrix along with various other metrics for the predictions. 
#The performance analysis of the GBM model:
confusionMatrix(predictions,testSet[,outcomeName])


