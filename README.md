# Credit Risk Analysis

## Credit Risk Overview:
In this portion of the project, I'll use different techniques to train and evaluate models. Using the credit card credit dataset from LendingClub, I'll oversample the data using the RandomOverSampler and SMOTE algorithms and undersample the data using the ClusterCentroids algorithm. Then, I’ll combine over- and undersampling using the SMOTEENN algorithm. Next, I'll compare two new machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, in an attempt to predict credit risk. Once completed, I'll evaluate the performance of these models and make a written recommendation on whether they should be used to predict credit risk.

## Credit Risk Results:

### Oversampling
  -  RandomOverSampler - 
     
     ![RandomOversampler_ConfusionMatrix](Images/RandomOversampler_ConfusionMatrix.PNG)
     -   Balanced Accuracy Score - 0.6663237827524566
     -   Precision - Low Risk seems very reliable, but low risk does not
     -   Recall - Decent likelhood of correctness at .70 and .63
 
 -  SMOTE - 
     
     ![SMOTE_ConfustionMatrix](Images/SMOTE_ConfustionMatrix.PNG) 
     - balanced accuracy score - 0.6623064259185507  
     - precision
     - recall

### Undersampling
 - ClusterCentroids - 
 
    ![ClusterCentroids_ConfustionMatrix](Images/ClusterCentroids_ConfustionMatrix.PNG) 
    - balanced accuracy score - 0.6623064259185507  
    - precision - Reliable positive classification for low risk, but unrealiable at .01 fir high risk
    - recall- low comparitively speaking (.40) for low-risk (majority class) but decent for high risk

### Over- and Undersampling
  
  - SMOTEENN - 
  
    ![SMOTEENN_ConfustionMatrix](Images/SMOTEENN_ConfustionMatrix.PNG)
    - balanced accuracy score - 0.6447701423556762   
    - precision very reliable for low risk, low for low risk
    - recall  - relatively high percentages for both so low risk .57 and high risk .72

### Ensemble Classifiers
  
  - BalancedRandomForestClassifier - 
  
    ![BalancedRandomForestClassifier_ConfustionMatrix](Images/BalancedRandomForestClassifier_ConfustionMatrix.PNG)
    - balanced accuracy score  - 0.7885466545953005 
    - precision - reliable for low risk , a bit higher on high risk than others
    - recall - catorized correctly at low risk 87 and high risk .70
  
  - EasyEnsembleClassifier - 
  
   ![EasyEnsembleClassifier_ConfustionMatrix](Images/EasyEnsembleClassifier_ConfustionMatrix.PNG)
    
   - balanced accuracy score  - 0.9322447299687874
   - precision reliablilty is higher for high risk than any other .09, and still reliable for the majority class low risk at 1
   - recall - it seems that the sensitivity is very high for this with .92 high risk (far higher than others, and .94 for low risk.


## Credit Risk Summary: 
The results of many of the models were very similiar with slight differences in numbers.  I would recommend the EasyEnsembleClassifier because of the higher balanced accuracy score, higher precision score for the minority class, and higher sensitivity for both high and low risk.each model learns from the errors of the previous model, and the models form an ensemble:AdaBoost, is easy to understand. In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized:
Run efficiently on large datasets.
