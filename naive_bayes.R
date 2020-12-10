naive_bayes <- function(training.dataset, test.dataset){  
  ##read data from file to a data frame
  training.table <- trainSparse
  test.table <- testSparse
  
  ##retrieve class and features from training data
  training.class <- training.table[, 200]
  training.features <- training.table[,-200]
  remove(training.table)
  
  ##funciton for calculating priors
  calculate.priors <- function(class.vector){
    priors <- c()
    for (class in unique(class.vector)){
      priors <- rbind(priors, c(class, length(class.vector[class.vector==class])/length(class.vector)))
      colnames(priors) <- c("classification", "probability")
    }
    return (priors)
  }
  priors <- calculate.priors(training.class)
  
  ##Learn the features by calculating likelihood
  likelihood.list <- list()
  #calculate CPD by feature
  for (i in 1:dim(training.features)[2]){
    feature.values <- training.features[, i]
    unique.feature.values <- unique(feature.values)
    likelihood.matrix <- matrix(rep(NA), nrow=dim(priors)[1], ncol=length(unique.feature.values))
    colnames(likelihood.matrix) <- unique.feature.values
    rownames(likelihood.matrix) <- priors[, "classification"]
    for (item in unique.feature.values){
      likelihood.item <- vector()
      for (class in priors[, "classification"]){
        feature.value.inclass <- feature.values[training.class==class]
        likelihood.value <- length(feature.value.inclass[feature.value.inclass==item])/length(feature.value.inclass)
        likelihood.item <- c(likelihood.item, likelihood.value)
      }
      likelihood.matrix[, item] <- likelihood.item
    }
    likelihood.list[[i]] <- likelihood.matrix
  }
  
  
  ##Predict class for the test dataset
  #retrieve the features and target class of the testing dataset
  test.features <- test.table[, -200]
  test.target.class <- test.table[, 200]
  test.predict.class <- rep(NA, length(test.target.class))
  remove(test.table)
  
  #calculate posteriors for each test data record
  for (i in 1:dim(test.features)[1]){
    record <- test.features[i, ]
    posterior <- vector()
    #calculate posteriors for each possible class of that record
    for (class in priors[, "classification"]){
      #initialize posterior as the prior value of that class
      posterior.value <- as.numeric(priors[priors[, "classification"]==class, 2])
      likelihood.v <- c()
      for (item in 1:length(record)){
        likelihood.value <- likelihood.list[[item]][class, as.character(record[1, item])]
        likelihood.v <- c(likelihood.v, likelihood.value)
        posterior.value <- as.numeric(posterior.value) * as.numeric(likelihood.value)
      }
      posterior <- rbind(posterior, c(class, posterior.value)) 
    }
    predict.class <- posterior[posterior[,2]==max(as.numeric(posterior[,2])),1]
    test.predict.class[i] <- predict.class
  }
  accuracy <- length(test.predict.class[test.predict.class==test.target.class])/length(test.target.class)
  missclassification <- 1 - accuracy
  
  print(paste("Accuracy:", accuracy))
  print(paste("\nMissclassification:", missclassification))
}

# Function Call - Make sure to change path to location on your computer!
naive_bayes("A:\\Alex\\Documents\\UCCS 2020\\Fall 2020\\Ind Study - Machine Learning\\Project 2\\20_newsgroups_Train.csv", "A:\\Alex\\Documents\\UCCS 2020\\Fall 2020\\Ind Study - Machine Learning\\Project 2\\20_newsgroups_Test.csv")
