# Credit Risk Analysis

## Credit Risk Overview:
In this portion of the project, I'll use different techniques to train and evaluate models. Using the credit card credit dataset from LendingClub, I'll oversample the data using the RandomOverSampler and SMOTE algorithms and undersample the data using the ClusterCentroids algorithm. Then, I’ll combine over- and undersampling using the SMOTEENN algorithm. Next, I'll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, in an attempt to predict credit risk. Once completed, I'll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Credit Risk Results:

### Oversampling
The idea is simple and intuitive: If one class has too few instances in the training set, we choose more instances from that class for training until it's larger.
  
  -  RandomOverSampler - 17.10.1instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. The accuracy score is high at around 90%, but this number can be misleading, especially in an unbalanced dataset.While precision ("pre" column) and recall ("rec" column) are high for the majority class, precision is low for the minority class.
     -  balanced accuracy score - 0.6663237827524566
     -  precision - To summarize, in machine learning, precision is a measure of how reliable a positive classification is. The following formulation may help you in remembering precision: "I know that the test for cancer came back positive. How likely is it that I have cancer?
     -  recall+ - I know that I have cancer. How likely is it that the test will diagnose it?"Sensitivity is a measure of how many people who actually have cancer were correctly diagnosed.
     -  ![RandomOversampler_ConfusionMatrix](Images/RandomOversampler_ConfusionMatrix.PNG)
     -   Balanced Accuracy Score - 0.6663237827524566
     -   Precision - Low Risk seems very reliable, but low risk does not
     -   Recall - Decent likelhood of correctness at .70 and .63
 
 -  SMOTE - In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling. Another deficiency of SMOTE is its vulnerability to outliers. We said earlier that a minority class instance is selected, and new values are generated based on its distance from its neighbors. If the neighbors are extreme outliers, the new values will reflect this. Finally, keep in mind that sampling techniques cannot overcome the deficiencies of the original dataset!
     - balanced accuracy score - 0.6623064259185507  
     - precision and recall
     - ![SMOTE_ConfustionMatrix](Images/SMOTE_ConfustionMatrix.PNG) 
     - balanced accuracy score - 0.6623064259185507  
     - precision
     - recall

### Undersampling
Undersampling takes the opposite approach of oversampling. Instead of increasing the number of the minority class, the size of the majority class is decreased.Keep in mind that both oversampling and undersampling involve tradeoffs. Oversampling addresses class imbalance by duplicating or mimicking existing data. In contrast, undersampling only uses actual data. On the other hand, undersampling involves loss of data from the majority class. Furthermore, undersampling is practical only when there is enough data in the training set. There must be enough usable data in the undersampled majority class for a model to be useful.
 
 - ClusterCentroids - Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.While resampling can attempt to address imbalance, it does not guarantee better results.
    - balanced accuracy score - 0.6623064259185507  
    - precision
    - recall- 
    - ![ClusterCentroids_ConfustionMatrix](Images/ClusterCentroids_ConfustionMatrix.PNG) 
    - balanced accuracy score - 0.6623064259185507  
    - precision - Reliable positive classification for low risk, but unrealiable at .01 fir high risk
    - recall- low comparitively speaking (.40) for low-risk (majority class) but decent for high risk

### Over- and Undersampling
  
  - SMOTEENN - SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process: Oversample the minority class with SMOTE and Clean the resulting data with an undersampling strategyemoving some of each class's outliers from the dataset. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
    - balanced accuracy score - 0.6447701423556762   
    - precision
    - recall 
    - ![SMOTEENN_ConfustionMatrix](Images/SMOTEENN_ConfustionMatrix.PNG)
    - balanced accuracy score - 0.6447701423556762   
    - precision very reliable for low risk, low for low risk
    - recall  - relatively high percentages for both so low risk .57 and high risk .72

### Ensemble Classifiers
  
  - BalancedRandomForestClassifier - These simple trees are weak learners because they are created by randomly sampling the data and creating a decision tree for only that small portion of data. And since they are trained on a small piece of the original data, they are only slightly better than a random guess. However, many slightly better than average small decision trees can be combined to create a strong learner, which has much better decision making power.
Random forest algorithms are beneficial because they:

Are robust against overfitting as all of those weak learners are trained on different pieces of the data.
Can be used to rank the importance of input variables in a natural way.
Can handle thousands of input variables without variable deletion.
Are robust to outliers and nonlinear data.
    - balanced accuracy score  - 0.7885466545953005 
    - precision
    - recall
    ![BalancedRandomForestClassifier_ConfustionMatrix](Images/BalancedRandomForestClassifier_ConfustionMatrix.PNG)
    - balanced accuracy score  - 0.7885466545953005 
    - precision - reliable for low risk , a bit higher on high risk than others
    - recall - catorized correctly at low risk 87 and high risk .70
  
  - EasyEnsembleClassifier - In boosting, however, the weak learners are not combined at the same time. Instead, they are used sequentially, as one model learns from the mistakes of the previous model.Like bagging, boosting is also a technique to combine a set of weak learners into a strong learner. We saw in bagging that the different models work independently of one another. In contrast, boosting trains a sequence of weak models. As shown below, each model learns from the errors of the previous model, and the models form an ensemble:AdaBoost, is easy to understand. In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized:
Run efficiently on large datasets.
   
   ![EasyEnsembleClassifier_ConfustionMatrix](Images/EasyEnsembleClassifier_ConfustionMatrix.PNG)
    
   - balanced accuracy score  - 0.9322447299687874
   - precision reliablilty is higher for high risk than any other .09, and still reliable for the majority class low risk at 1
   - recall - it seems that the sensitivity is very high for this with .92 high risk (far higher than others, and .94 for low risk.
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.There is a bulleted list that describes the balanced accuracy score and the precision and recall scores of all six machine learning models

## Credit Risk Summary: 
The results of many of the models were very similiar with slight differences in numbers.  I would recommend the EasyEnsembleClassifier because of the higher balanced accuracy score, higher precision score for the minority class, and higher sensitivity for both high and low risk.
