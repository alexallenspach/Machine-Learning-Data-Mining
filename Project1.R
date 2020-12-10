#***************************************************************************************************
# Student: Alex Allenspach
# Short Description: In this task, for each of the datasets above create a training dataset
#                    and test dataset. Make the number of instances in training and test 
#                    dataset be 80% and 20% of the original dataset. Implement a function
#                    executable file, that uses Decision Trees, Random Forest, and Naive 
#                    Bayes to train a model, and then applies the model to classify your 
#                    test data. 
#***************************************************************************************************


#Libraries:
library(readr)
library(caTools)
library(caret)
library(e1071)
library(tictoc)
library(rpart)
library(randomForest)


#***************************************************************************************************
# TASK 1


# Remove "?" from continuous variables so they are not perceived as factor variables
# There is option to replace "?" with NA but then predictions cannot be done with algorithms
# NOTE: There are 32,561 instances for census since no "?" are in continuous variables
#       but there are 666 instances for credit (original 690) because 24 instances have
#       "?" in a continuous variable thus are removed

adult <- na.omit(read_csv("adult.data", col_names = FALSE, col_types = cols(X1 = col_double(),X3 = col_double(),X5 = col_double(),X11 = col_double(),X12 = col_double(), X13 = col_double())))
crx <- na.omit(read_csv("crx.data", col_names = FALSE, col_types = cols(X2 = col_double(),X3 = col_double(),X8 = col_double(),X11 = col_double(),X14 = col_double(), X15 = col_double())))


#***************************************************************************************************
# TASK 2


# NOTE: It is assumed that any text files with data can be used as arguments but certain 
#       requirements should be followed and some preliminary processing should be done if it 
#       is needed. In particular, the dependent variable should be called “dependent”. 
#       In turn, there are not any limitations in relation to names and number of explanatory 
#       variables. Another requirement is that all predictors should be factorized.

# Census Income Dataset dependent variable
names(adult)[names(adult) == "X15"] <- "dependent" 
# Credit Approval Dataset dependent variable
names(crx)[names(crx) == "X16"] <- "dependent" 
# Random Sampling without replacement for Census Income - 80% (train), 20% (test)
divider1 <- sample.split(adult$dependent, SplitRatio = 0.8) 
census_trainset <- subset(adult, divider1 == TRUE)
census_testset <- subset(adult, divider1 == FALSE)
# Save Training file for CI
write.table(census_trainset, file = "census_trainset.txt", row.names=FALSE, quote = FALSE)
# Save Test file for CI
write.table(census_testset, file = "census_testset.txt", row.names=FALSE, quote = FALSE)
# Random Sampling without replacement for Credit Approval - 80% (train), 20% (test)
divider2 <- sample.split(crx$dependent, SplitRatio = 0.8)
credit_trainset <- subset(crx, divider2 == TRUE)
credit_testset <- subset(crx, divider2 == FALSE)
# Save Training file for CA
write.table(credit_trainset, file = "credit_trainset.txt", row.names=FALSE, quote = FALSE)
# Save Test file for CA
write.table(credit_testset, file = "credit_testset.txt", row.names=FALSE, quote = FALSE)


#***************************************************************************************************
# TASK 3, TASK 4, TASK 5

# NOTE: For Classification - continuous variables are recoded to 0 (if the value is lower 
#                            than median) and 1 (if the value is higher than median).

# Decision Tree Model
decision_tree<-function(training_file, test_file)
{
  tic() # Start Stopwatch Timer
  # Download data inside function
  training_data<-read.table(file = training_file, header = TRUE, sep = " ")
  test_data<-read.table(file = test_file, header = TRUE, sep = " ")
  # Build classification decision tree model using rpart library
  DTModel<-rpart(dependent~.,data=training_data,method="class", minbucket = 20)
  # Classification
  test_data$object_ID<-rownames(test_data)
  test_data$predicted_class<-predict(DTModel,newdata=test_data, type = 'class')
  test_data$true_class<-factor(test_data$dependent)
  test_data$accuracy<-ifelse(test_data$predicted_class==test_data$true_class,1,0)
  # Confusion Matrix
  confMatrix <- confusionMatrix(test_data$dependent, test_data$predicted_class)
  cat("\nDECISION TREE\n")
  print(test_data[c("object_ID", "predicted_class", "true_class", "accuracy")])
  print(confusionMatrix(test_data$predicted_class,test_data$true_class)$overall)
  print(confusionMatrix(test_data$predicted_class,test_data$true_class)$byClass)
  print(confMatrix)
  toc() # End Stopwatch Timer
  cat("\n")
}

# Random Forest Model
random_forest<-function(training_file, test_file)
{
  tic() # Start Stopwatch
  # Download data inside function
  training_data<-read.table(file = training_file, header = TRUE, sep = " ")
  test_data<-read.table(file = test_file, header = TRUE, sep = " ")
  # Build Random Forest using randomForest library
  RFModel<-randomForest(dependent~.,data=training_data,ntree=500, mtry=4)
  # Equalize classes of training and test set
  xtest <- rbind(training_data[1, ] , test_data)
  xtest <- xtest[-1,]
  # Classification
  test_data$object_ID<-rownames(test_data)
  test_data$predicted_class<-predict(RFModel,newdata=xtest)
  test_data$true_class<-factor(test_data$dependent)
  test_data$accuracy<-ifelse(test_data$predicted_class==test_data$true_class,1,0)
  # Confusion Matrix
  confMatrix <- confusionMatrix(test_data$dependent, test_data$predicted_class)
  cat("\nRANDOM FOREST\n")
  print(test_data[c("object_ID", "predicted_class", "true_class", "accuracy")])
  print(confusionMatrix(test_data$predicted_class,test_data$true_class)$overall)
  print(confusionMatrix(test_data$predicted_class,test_data$true_class)$byClass)
  print(confMatrix)
  toc() # End Stopwatch Timer
  cat("\n")
}

# Naive Bayes Model
naive_bayes<-function(training_file, test_file)
{
  tic() # Start Stopwatch Timer
  # Download data inside function
  training_data<-read.table(file = training_file, header = TRUE, sep = " ")
  test_data<-read.table(file = test_file, header = TRUE, sep = " ")
  # Build Naive Bayes Model using e1071 library
  NBModel<-naiveBayes(dependent~.,data=training_data)
  # Classification
  test_data$object_ID<-rownames(test_data)
  test_data$predicted_class<-predict(NBModel,newdata=test_data)
  test_data$true_class<-factor(test_data$dependent)
  test_data$accuracy<-ifelse(test_data$predicted_class==test_data$true_class,1,0)
  # Confusion Matrix
  confMatrix <- confusionMatrix(test_data$dependent, test_data$predicted_class)
  cat("\nNAIVE BAYES\n")
  print(test_data[c("object_ID", "predicted_class", "true_class", "accuracy")])
  print(confusionMatrix(test_data$predicted_class,test_data$true_class)$overall)
  print(confusionMatrix(test_data$predicted_class,test_data$true_class)$byClass)
  print(confMatrix)
  toc() # End Stopwatch Timer
  cat("\n")
}

# Main Function!
DTvsRFvsNB<-function(training_file, test_file)
{
  decision_tree(training_file, test_file)
  random_forest(training_file, test_file)
  naive_bayes(training_file, test_file)
}


#***************************************************************************************************
# TASK 6


# Handle Missing Values - By mean value for each continuous variable
crx2 <- read_csv("crx.data", col_names = FALSE, col_types = cols(X2 = col_double(),X3 = col_double(),X8 = col_double(),X11 = col_double(),X14 = col_double(), X15 = col_double()))
crx2$X2<-ifelse(is.na(crx2$X2),mean(na.omit(crx2$X2)),crx2$X2)
crx2$X3<-ifelse(is.na(crx2$X3),mean(crx2$X3),crx2$X3)
crx2$X8<-ifelse(is.na(crx2$X8),mean(crx2$X8),crx2$X8)
crx2$X11<-ifelse(is.na(crx2$X11),mean(crx2$X11),crx2$X11)
crx2$X14<-ifelse(is.na(crx2$X14),mean(na.omit(crx2$X14)),crx2$X14)
crx2$X15<-ifelse(is.na(crx2$X15),mean(crx2$X15),crx2$X15)
# Credit Approval Task 6 Dataset dependent variable
names(crx2)[names(crx2) == "X16"] <- "dependent"
# Random Sampling without replacement for Credit Approval - 80% (train), 20% (test)
divider3 <- sample.split(crx2$dependent, SplitRatio = 0.8)
credit_trainset2 <- subset(crx2, divider3 == TRUE)
credit_testset2 <- subset(crx2, divider3 == FALSE)
# Save Training file for CA Task 6
write.table(credit_trainset2, file = "Task6_credit_trainset.txt", row.names=FALSE, quote = FALSE)
# Save Test file for CA Task 6
write.table(credit_testset2, file = "Task6_credit_testset.txt", row.names=FALSE, quote = FALSE)


# MAIN EXECUTION
DTvsRFvsNB("credit_trainset.txt","credit_testset.txt")
DTvsRFvsNB("census_trainset.txt","census_testset.txt")
DTvsRFvsNB("Task6_credit_trainset.txt","Task6_credit_testset.txt")
