---
title: "Human Activity Recognition"
author: "G Arango"
date: "Saturday, September 26, 2015"
output: html_document
---
# Acknowlegments

The training data and the ideas used in this project came from:
"Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013."

# Summary

People regularly quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

We will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 
They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
The goal to predict the manner in which they did the exercise. The "classe" variable in the training set represent five different fashions for the activity: exactly according to the specification (A), throwing the elbows to the front (B), lifting the dumbbell only halfway (C), 
lowering the dumbbell only halfway (D) and throwing the hips to the front (E).

The prediction model will be used to predict 20 different test cases. 

The training data contains 159 possible predictors.


# Data loading and cleaning

We load the prediction data and we exclude all columns in the dataframe having only NA or null values because they can be used for prediction. We ignore also X, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window with have no relation with the human activity.

Then we do the same from the trainig, we exclude all those non predicting columns to simplify the model.


```r
predictset = read.csv("pml-testing.csv", header=TRUE, na.strings=c("NA",""))
#ignoring columns X raw_timestamp_part_1 raw_timestamp_part_2 cvtd_timestamp new_window num_window
predictset = predictset[,c(2,9:ncol(predictset)-1)]
predictset = predictset[,colSums(is.na(predictset))<nrow(predictset)]
predictset$classe=as.factor(rep(c("A","B","C","D","E"),4))

sampleData = read.csv("pml-training.csv", header=TRUE,na.strings=c("NA",""))
dim(sampleData)
```

```
## [1] 19622   160
```

```r
sampleData = sampleData[names(predictset)]
sampleData$classe=factor(sampleData$classe)
predictset$classe=as.factor(rep(c("A","B","C","D","E"),4))
dim(sampleData)
```

```
## [1] 19622    54
```

***From an initial set of 160 columns with have excludednore than 100.***

# Model computation 

The training will be partitionned on a .6 for training and .4 for testing
We train classification regression models using caret package for rainforest. 

To be able to predict and use a confusion matrix to compute the error), the train method uses a default trainControl which includes the cross-validation method "cv"


```r
library(caret)
library(randomForest)
inTrain = createDataPartition(sampleData$classe, p = .6)[[1]]
training = sampleData[ inTrain,]
testing = sampleData[-inTrain,]
nrow(training);nrow(testing)
```

```
## [1] 11776
```

```
## [1] 7846
```

```r
set.seed(666)
modelRF <- train(classe ~ ., data = training, method="rf", importance=T)
print(modelRF)
```

```
## Random Forest 
## 
## 11776 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9846879  0.9806194  0.002643298  0.003334136
##   29    0.9866007  0.9830414  0.001807704  0.002283332
##   57    0.9806082  0.9754599  0.004332355  0.005470354
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```

# Results

The accuracy of the prediction model for mtry = 29 is almost 100%

We compute the confusion matrix for the testing set


```r
predRF = predict(modelRF,testing)
cmRF = confusionMatrix(testing$classe, predRF)
cmRF$overall['Accuracy']
```

```
##  Accuracy 
## 0.9912057
```

```r
cmRF$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2226    5    0    1    0
##          B   20 1493    5    0    0
##          C    0    8 1355    5    0
##          D    0    0   17 1268    1
##          E    0    0    2    5 1435
```

Confusion matrix shows that the testing set classification with few missclassifications.

# "classe" values results for the prediction set


```r
predsetclasse = predictset[names(predictset)!="problem_id"]
res=predict(modelRF,newdata=predsetclasse)
data.frame(res)
```

```
##    res
## 1    B
## 2    A
## 3    B
## 4    A
## 5    A
## 6    E
## 7    D
## 8    B
## 9    A
## 10   A
## 11   B
## 12   C
## 13   B
## 14   A
## 15   E
## 16   E
## 17   A
## 18   B
## 19   B
## 20   B
```
