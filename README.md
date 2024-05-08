# Machine Learning Classifications
Using various supervised ML models to compare best classification 

## Dataset
- Viewer Engagement Data sourced from:  
https://github.com/sahanbull/VLE-Dataset?tab=readme-ov-file#vle-datasets
- 7 features (title_word_count, document_entropy, freshness, easiness, fraction_stopword_presence, normalization_rate, speaker_speed, silent_period_rate)
- Binary boolean label indicating Engaged or not 
- Training dataset of 9239 rows with inputs and labels for training 
- Testing dataset of 2309 rows with only inputs 

## Approach
Firstly, we turned a regression problem into a classification one, since the original target was Median Viewing Time.  
Done by defining the label True if Median Viewing Time was >30% of total duration, else False.  
Often reduced dimensionality is more useful, e.g. engaged or not instead of exactly the median view time.  
Classification also allows us to compare more ML models, e.g. Naïve Bayes, Random Forest, etc.   

### Supervised ML Classifiers Used:
- Gaussian Naïve Bayes
- kNN classifier
- Logit
- Support Vector Machines 
- Neural Networks (MLP)
- Decision Trees
- Ensemble-Trees (Random Forests)
- Gradient-Boosted Decision Trees

### Evaluation Metric
Once we trained the models, we use scikit-learn's GridSearchCV() to optimize parameters based on roc-auc  
Basically 2 optimizations are to be done:  
- The GridSearchCV().best_estimator_ attribute finds estimator's best parameters vis a vis roc-auc
- We define find_best_model() to select the best of the best, best estimator among the estimator 'finalists'

We use this model to predict the training dataset 
