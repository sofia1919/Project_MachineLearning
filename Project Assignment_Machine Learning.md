Project Assignment Machine Learning
========================================================

**Background**

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

**Goal of the Project assignment**

The goal of our project is to predict the manner in which 6 participants did the exercise.(this is the "classe" variable in the training set.)We should also use our prediction model to predict 20 different test cases. We may use any of the other variables to predict with. At the end we should create a report describing how we built our model, how we used cross validation, what we think the expected out of sample error is, and why we made the choices we did. 

**Report**

* Reading the train data set
* Removing all the unneccessary columns containing NAs,Div#1, "blanks"and data we don't need.    Since we should use only the data related to the outcome in the scope of the study is a HAR (Human activity recognition study) .
* Changing the data format for all the predictors to numeric, except outcome "classe" (54th  column).It remains class="factor"
* Splitting the data set into training and validation set (70%:30%)
* Setting a seed (12345)
* Training the train data set with Random Forest and 5-fold cross validation (method="rf" and train controller=method="cv",number=5)
 I choose this model because of its Accuracy=0.997 and since it was faster than random forest method with method "oob"-out-of-bag" and in this way I could use a cross validation to estimate the out-of sample error. Also I've choosen Random forests because they can output a list of predictor variables that they believe to be important in predicting the outcome.
* Predicting the validation set with the same model.
* Calculating the ConfusionMatrix between predicted validation set and the actual validation$classe variable
* Calculating the in-sample error for the trained data set (1- Accuracy) and out-of-sample error for the validation set (1- Accuracy).
* Reading the test data set
* Removing all the unneccessary columns containing NAs,Div#1, "blanks"and data we don't need.
* Changing the data format for all the predictors to numeric (except the54th column) .
* Using the training model fit on the testdata to predict the 20 test cases
* Prediction Output for the 20 test cases



```r
setwd("C:/Users/user/Desktop/Coursera/Practical Machine Learning/Project")

#Reading the training data 

RawData<-read.csv("pml-training.csv")

#Removing all the unneccessary columns containing NAs,Div#1 und data we don't need
traindata<-RawData[ ,!grepl("X|user_name|timestamp|new_window|max|min|amplitude|var|stddev|avg|kurtosis|skewness",names(RawData))]

#Changing the data format for all the predictors to numeric, outcome "classe" (54th column) remains class=factor
cols<-c(1:ncol(traindata)!=54)  
traindata[,cols]<-as.numeric(unlist(traindata[,cols]))

library(lattice)
library(ggplot2)
library(caret)
BuildModel<-createDataPartition(y=traindata$classe, p=0.7, list=FALSE)
training<-traindata[BuildModel,]
validation<-traindata[-BuildModel,]

set.seed(12345)
library(randomForest)

#Training the model with Random Forest and 5-folds cross validation
modelFit1<-train(training$classe~., method="rf", trControl=trainControl(method="cv",number=5),allowParallel=TRUE,data=training)

## modelFit1

## Random Forest 
##
## 13737 samples

##   53 predictors

##    5 classes: 'A', 'B', 'C', 'D', 'E' 

## No pre-processing

## Resampling: Cross-Validated (5 fold) 

## Summary of sample sizes: 10987, 10991, 10989, 10991, 10990 

## Resampling results across tuning parameters:

##  mtry | Accuracy | Kappa | Accuracy SD | Kappa SD
---------|----------|-------|-------------|----------

##   2  |   0.994   | 0.992 | 0.00225     |  0.00285 

##  27  |   0.997   | 0.996 | 0.00149     |  0.00189 

##  53  |  0.992    | 0.99  | 0.00271     |  0.00343 


## Accuracy was used to select the optimal model using  the largest value.

## The final value used for the model was mtry = 27. 

##modelFit1$finalModel

##Call:

##randomForest(x = x, y = y, mtry = param$mtry, allowParallel = TRUE) 

##               Type of random forest: classification

##                    Number of trees: 500

##No. of variables tried at each split: 27
##
##        OOB estimate of  error rate: 0.25%

##Confusion matrix:

##     A   | B   | C |   D  |  E  |class.error
-----------|-----|---|------|-----|-----------

##  A |3905|  0  | 0  |  0   | 1   | 0.0002560164

##  B |   4| 2650|  3 |   1  | 0   |  0.0030097818

##  C |   0|   5 | 2390|  1  |  0  |  0.0025041736

##  D |  0 |  0  |  11 |2240 |  1  |  0.0053285968

##  E |  0 |   1 |   0  |  7 |2517 | 0.0031683168

#Predicting the validation set with the same model.
predict_validation<-predict(modelFit1,validation)

# Calculating the ConfusionMatrix between predicted validation set and the actual validation$classe variable
##confusionMatrix(validation$classe,predict_validation)

##Confusion Matrix and Statistics

##          Reference

##Prediction    A    B    C    D    E

##         A 1673    1    0    0    0

##         B    3 1135    1    0    0

##         C    0    2 1024    0    0

##         D    0    0    3  961    0

##         E    0    0    0    4 1078
##
##Overall Statistics
##                                         
##              Accuracy : 0.9976     

##                 95% CI : (0.996, 0.9987)

##    No Information Rate : 0.2848    

##    P-Value [Acc > NIR] : < 2.2e-16      
##                                         
##                  Kappa : 0.997     

## Mcnemar's Test P-Value : NA             
##
##Statistics by Class:
##
##                     Class: A Class: B Class: C Class: D Class: E

##Sensitivity            0.9982   0.9974   0.9961   0.9959   1.0000

##Specificity            0.9998   0.9992   0.9996   0.9994   0.9992

##Pos Pred Value         0.9994   0.9965   0.9981   0.9969   0.9963

##Neg Pred Value         0.9993   0.9994   0.9992   0.9992   1.0000

##Prevalence             0.2848   0.1934   0.1747   0.1640   0.1832

##Detection Rate         0.2843   0.1929   0.1740   0.1633   0.1832

##Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839

##Balanced Accuracy      0.9990   0.9983   0.9978   0.9976   0.9996

#Calculating the in-sample error for the trained data set (1- Accuracy) and out-of-sample error for the validation set (1- Accuracy)

#In-Sample Error: 1- Accuracy(ModelFit1(on training set)=1-0.997=0.003=0.3%

#Out-of-sample Error: 1- Accuracy(Prediction on validation set)=1-0.997=0.003=0.3%

#Reading the test data set
RawData_test<-read.csv("pml-testing.csv")

#Removing all the unneccessary columns containing NAs,Div#1, "blanks"and data we don't need.
testdata<-RawData_test[ ,!grepl("X|user_name|timestamp|new_window|max|min|amplitude|var|stddev|avg|kurtosis|skewness",names(RawData_test))]
 
#Changing all predictor variables to class=numeric, all columns except 54th column
cols<-c(1:ncol(testdata)!=54)
testdata[,cols]<-as.numeric(unlist(testdata[,cols]))

#Predicting the 20 test cases based on the train model for the training data
predicttest<-predict(modelFit,testdata)
#predicttest
# #   B A B A A E D B A A B C B A E E A B B B
```


