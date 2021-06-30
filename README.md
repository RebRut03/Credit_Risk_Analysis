# Credit Risk Analysis

## Credit Risk Overview:
In this portion of the project, I'll use different techniques to train and evaluate models. Using the credit card credit dataset from LendingClub, I'll oversample the data using the RandomOverSampler and SMOTE algorithms and undersample the data using the ClusterCentroids algorithm. Then, I’ll combine over- and undersampling using the SMOTEENN algorithm. Next, I'll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, in an attempt to predict credit risk. Once completed, I'll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Credit Risk Results:

### Oversampling
The idea is simple and intuitive: If one class has too few instances in the training set, we choose more instances from that class for training until it's larger.
  
  -  RandomOverSampler - 17.10.1instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. The accuracy score is high at around 90%, but this number can be misleading, especially in an unbalanced dataset.While precision ("pre" column) and recall ("rec" column) are high for the majority class, precision is low for the minority class.
     -  balanced accuracy score   
     -  precision
     -  recall
 
 -  SMOTE - In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling. Another deficiency of SMOTE is its vulnerability to outliers. We said earlier that a minority class instance is selected, and new values are generated based on its distance from its neighbors. If the neighbors are extreme outliers, the new values will reflect this. Finally, keep in mind that sampling techniques cannot overcome the deficiencies of the original dataset!
     - balanced accuracy score   
     - precision
     - recall  

### Undersampling
Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.Keep in mind that both oversampling and undersampling involve tradeoffs. Oversampling addresses class imbalance by duplicating or mimicking existing data. In contrast, undersampling only uses actual data. On the other hand, undersampling involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.
 
 - ClusterCentroids - Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.While resampling can attempt to address imbalance, it does not guarantee better results.
    - balanced accuracy score   
    - precision
    - recall- 

### Over- and Undersampling
  
  - SMOTEENN - SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process: Oversample the minority class with SMOTE and Clean the resulting data with an undersampling strategyemoving some of each class's outliers from the dataset. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
    - balanced accuracy score   
    - precision
    - recall 

### Ensemble Classifiers
  
  - BalancedRandomForestClassifier
    - balanced accuracy score   
    - precision
    - recall 
  
  - EasyEnsembleClassifier
    - balanced accuracy score   
    - precision
    - recall 
    
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models

## Credit Risk Summary: 
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.There is a summary of the results (2 pt)
There is a recommendation on which model to use, or there is no recommendation with a justification
