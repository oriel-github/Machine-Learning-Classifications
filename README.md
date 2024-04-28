# Machine-Learning-Classifications
Using various supervised ML models to compare best classification

## Metric
Find best ML classifier based on roc-auc 

## Evaluation Method
scikit-learn's GridSearchCV Validator

## Supervised ML models
- Logistic (with Polynomial features)
- Naïve Bayes
- Neural Networks (MLP)
- kNN classifier
- Ensemble-Trees (Random Forests)
- Gradient-Boosted Support Vector Machines (Linear/Kernel)

## Dataset
- Viewer Engagement Data sourced from https://github.com/sahanbull/VLE-Dataset?tab=readme-ov-file#vle-datasets
- 7 features (title_word_count, document_entropy, freshness, easiness, fraction_stopword_presence, normalization_rate, speaker_speed, silent_period_rate)
- Binary boolean label indicating Engaged or not (True if Median viewing time was >30% of total duration, else False)
- Training dataset with 9239 rows with inputs and labels for training 
- Testing dataset with 2309 rows with only inputs 
