# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a supervised binary classification model trained to predict whether an individual's salary is greater than $50K based on census data. The model was implemented using scikit-learn and trained with a RandomForestClassifier.

## Intended Use
The model is intended for educational purposes in the Udacity project and demonstrates how to build, evaluate, and deploy a machine learning inference pipeline. It should be used to predict salary class from the provided census features after the same preprocessing steps are applied.

## Training Data
The training data comes from the census dataset provided with the project. It includes both categorical and numerical features such as workclass, education, marital-status, occupation, relationship, race, sex, native-country, age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week. The target label is `salary`.

## Evaluation Data
The dataset was split into training and test sets using `train_test_split`. The test set was used to evaluate model performance after training.

## Metrics
The model was evaluated using precision, recall, and F1 score.

- Precision: 0.7419
- Recall: 0.6384
- F1 Score: 0.6863

These metrics indicate that the model performs reasonably well at identifying higher-income individuals, though recall is lower than precision.

## Ethical Considerations
This model is trained on census income data, which may reflect historical and societal biases. Features such as sex, race, and native-country may introduce fairness concerns if the model were used in a real decision-making setting. The model should not be used for high-stakes decisions such as hiring, lending, or admissions without careful bias analysis and governance.

## Caveats and Recommendations
This model was developed for a classroom project and is not production-ready for real-world decision-making. Performance may vary across demographic groups, so slice-based evaluation is important. Future improvements could include hyperparameter tuning, fairness analysis, and more robust validation.
