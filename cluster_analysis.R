#**************************************************************************************************************************************************************
# Student:         Alex Allenspach
# Professor:       Bill Michael
# Class:           Ind Study - Machine Learning
# Due Date:        10 December 2020
# Description:     Clustering involves finding similarities between data according to the characteristics found in the data and grouping similar data objects 
#                  into clusters. For this clustering project, you will analyze a study of Asian Religious and Biblical Texts Data Set. 
# Goal of Project: The goal is to use clustering algorithms to determine which documents should be in same group based on the document matrices. 
#                  Ultimately, you want document from same religious group in same cluster. 
#**************************************************************************************************************************************************************


install.packages(c("readr","factoextra", "fpc", "NbClust", "clusterSim", "tictoc", "quanteda"))

library(readr) 
library(factoextra)
library(fpc)
library(NbClust)
library(clusterSim)
library(tictoc)
library(quanteda)


# k-means clustering algorithm

kmeans_dm <- function(input_data, number_of_clusters) {
  
  tic()                                   # Start stopwatch
  religiousData <- read_csv(input_data)   # read in data
  
  # convert data frame into document feature matrix so we can use quanteda's built in tf-idf function
  data_DFM <- as.dfm(religiousData)
  
  # Apply tf-idf weight
  data_tf_idf <- dfm_tfidf(data_DFM)
  
  # convert back into data frame
  df <- as.data.frame(as.matrix(data_tf_idf))

  # K-means clustering
  kmeans_cluster <- eclust(df, "kmeans", k = number_of_clusters, nstart = 25, graph = FALSE, hc_metric = "euclidean")

  # Visualize k-means clusters
  plot <- fviz_cluster(kmeans_cluster, geom = "point", ellipse.type = "norm",
               palette = "jco", ggtheme = theme_minimal())

  # plot using SI score to determine optimal cluster number - shows 5 clusters is optimal
  #optimalCluster_plot <- fviz_nbclust(df, kmeans, method='silhouette')

  # Statistics for k-means clustering
  kmeans_stats <- cluster.stats(dist(df),  kmeans_cluster$cluster)
  kmeans_DBI <- index.DB(df, kmeans_cluster$cluster)

  # Print plots
  print(plot)
  #print(optimalCluster_plot)

  # Print DBI and SI score for specific cluster number
  cat("The DBI score for K-Means Clustering with", number_of_clusters, "clusters is:", kmeans_DBI$DB)
  cat("\nThe SI score for K-Means Clustering with", number_of_clusters, "clusters is:", kmeans_stats$avg.silwidth)
  cat("\n")
  toc()                       # Stop stopwatch
}



# k-medoids clustering algorithm

kmedoids_dm <- function(input_data, number_of_clusters) {
  
  tic()                                   # Start stopwatch
  religiousData <- read_csv(input_data)   # read in data
  
  # convert data frame into document feature matrix so we can use quanteda's built in tf-idf function
  data_DFM <- as.dfm(religiousData)
  
  # Apply tf-idf weight
  data_tf_idf <- dfm_tfidf(data_DFM)
  
  # convert back into data frame
  df <- as.data.frame(as.matrix(data_tf_idf))

  # K-medoids clustering
  kmedoids_cluster <- eclust(df, "pam", k = number_of_clusters, hc_metric = "euclidean", graph = FALSE)

  # Visualize k-medoids cluster
  plot <- fviz_cluster(kmedoids_cluster, show_labels = FALSE,
                       palette = "jco", as.ggplot = TRUE)

  # plot using SI score to determine optimal cluster number - shows 3 clusters is optimal
  #optimalCluster_plot <- fviz_nbclust(df, cluster::pam, method='silhouette')

  # Statistics for k-medoids clustering
  kmedoids_stats <- cluster.stats(dist(df),  kmedoids_cluster$cluster)
  kmedoids_DBI <- index.DB(df, kmedoids_cluster$cluster)

  # Print plots
  print(plot)
  #print(optimalCluster_plot)

  # Print DBI and SI score for specific cluster number
  cat("The DBI score for K-Medoids Clustering with", number_of_clusters, "clusters is:", kmedoids_DBI$DB)
  cat("\nThe SI score for K-Medoids Clustering with", number_of_clusters, "clusters is:", kmedoids_stats$avg.silwidth)
  cat("\n")
  toc()                       # Stop stopwatch
}



# hierarchical clustering algorithm

hierarchicalclustering_dm <- function(input_data, number_of_clusters) {
  
  tic()                                   # Start stopwatch
  religiousData <- read_csv(input_data)   # read in data
  
  # convert data frame into document feature matrix so we can use quanteda's built in tf-idf function
  data_DFM <- as.dfm(religiousData)
  
  # Apply tf-idf weight
  data_tf_idf <- dfm_tfidf(data_DFM)
  
  # convert back into data frame
  df <- as.data.frame(as.matrix(data_tf_idf))

  # Hierarchical clustering
  hierarchical_cluster <- eclust(df, "hclust", k = number_of_clusters, hc_metric = "euclidean",
                                 hc_method = "ward.D", graph = FALSE)

  # Visualize dendrograms
  plot <- fviz_dend(hierarchical_cluster, show_labels = FALSE,
                    palette = "jco", as.ggplot = TRUE)

  # plot using SI score to determine optimal clusater number - shows 3 clusters is optimal
  #optimalCluster_plot <- fviz_nbclust(df, hcut, k.max = 11, method='silhouette')

  # Statistics for hierarchical clustering
  hc_stats <- cluster.stats(dist(df),  hierarchical_cluster$cluster)
  hc_DBI <- index.DB(df, hierarchical_cluster$cluster)

  # Print plots
  print(plot)
  #print(optimalCluster_plot)

  # Print DBI and SI score for specific cluster number
  cat("The DBI score for Hierarchical Clustering with", number_of_clusters, "clusters is:", hc_DBI$DB)
  cat("\nThe SI score for Hierarchical Clustering with", number_of_clusters, "clusters is:", hc_stats$avg.silwidth)
  cat("\n")
  toc()                       # Stop stopwatch
}



# Main Function
cluster_analysis <- function(input_data, number_of_clusters) {
  kmeans_dm(input_data, number_of_clusters)
  kmedoids_dm(input_data, number_of_clusters)
  hierarchicalclustering_dm(input_data, number_of_clusters)
}

setwd("A:\\Alex\\Documents\\UCCS 2020\\Fall 2020\\Ind Study - Machine Learning\\Project 3")
  
# Main Execution
cluster_analysis("AllBooks_baseline_DTM_Unlabelled.csv", 2) # Make sure to change file path based on your file's location!

