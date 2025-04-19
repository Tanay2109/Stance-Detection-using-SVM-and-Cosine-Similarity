Fake News Detection using SVM

Loading the FNC1 dataset...

Dataset loaded successfully!

Number of bodies: 1683

Number of stances: 49972

Missing values in bodies: Body ID        0

articleBody    0

dtype: int64

Missing values in stances: Headline    0

Body ID     0

Stance      0

dtype: int64

Stance distribution:

Stance

unrelated    36545

discuss       8909

agree         3678

disagree       840

Name: count, dtype: int64

Stance distribution visualization saved as 'stance_distribution.png'

Merging datasets...

Preprocessing data...

Extracting features using TF-IDF...

Calculating similarity features...

Feature matrix shape: (49972, 10002)

Target variable shape: (49972,)

Splitting data into training and testing sets...

Training set size: 39977

Testing set size: 9995

Training the SVM model...

Making predictions...

Accuracy: 0.9704

Classification Report:
              precision    recall  f1-score   support

       agree       0.87      0.85      0.86       736
    disagree       0.69      0.64      0.66       168
     discuss       0.94      0.95      0.95      1782
   unrelated       0.99      1.00      0.99      7309

    accuracy                           0.97      9995
   macro avg       0.87      0.86      0.87      9995
weighted avg       0.97      0.97      0.97      9995

Confusion matrix visualization saved as 'confusion_matrix.png'

Performing hyperparameter tuning (this may take a while)...

Fitting 3 folds for each of 12 candidates, totalling 36 fits

Best parameters: {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}

Best cross-validation score: 0.9700

Test accuracy with best parameters: 0.9762

FNC-1 weighted score: 0.9428

Saving model and preprocessing components...

Model and preprocessing components saved successfully!

Testing the prediction function with a sample:

Headline: Climate change is a hoax

Body: Scientists worldwide agree that climate change is real and caused by human activities.

Predicted stance: agree

Fake News Detection model training and evaluation completed successfully!
