from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics

# Strarting the SparkSession
spark = SparkSession.builder.appName('cancer_diagnosis').getOrCreate()

# Loading the data from the .csv file
df = spark.read.csv('/Users/rijadkasumi/Desktop/project3/project3_data.csv', header = True, inferSchema = True)

# Convert the diagnosis column from categorical to numerical because of B Bening and M Malignant
indexer = StringIndexer(inputCol="diagnosis", outputCol="label")
df = indexer.fit(df).transform(df)

# VectorAssembler transforms the data by collecting multiple columns of your DataFrame and putting them into a single vector column. 
# This is necessary because each machine learning algorithm in Spark MLlib requires the input data to be in this format.
assembler = VectorAssembler(
    inputCols=['Radius_mean', 'Texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
               'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
               'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
               'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
               'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
               'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
               'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'],
    outputCol="features")

# Split the data into training, validation and testing sets
# random state seed = // accuracy may vary,important thing is to use the same seed value if you want to reproduce the same splits.
train_data, temp_data = df.randomSplit([0.6, 0.4], seed = 2023)
validation_data, test_data = temp_data.randomSplit([0.5, 0.5], seed = 2023)


# Define the Logistic Regression model using MLlib
# Limitin 10 iterations
lr = LogisticRegression(maxIter=10)

# Define the Random Forest model using MLlib
rf = RandomForestClassifier()

# ParamGrid for Cross Validation - GRID SEARCH

# Logistic Regression Parameter Grid -helps specify the grid of hyperparameters
# regParam determines the regularization strength- to avoid overlifting
# fit intercept decided wether to fit and intercet in the model
parameterGrid_lr = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .build()
# Random Forest Parameter Grid 
# numTrees 10,30 specifies the number of trees in the RF model- possible values of 10 and 30
parameterGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 30]) \
    .addGrid(rf.maxDepth, [5, 10]) \
    .build()

# Define the pipeline
# In Spark MLlib, a Pipeline is a sequence of data processing stages
pipeline_lr = Pipeline(stages=[assembler, lr])
pipeline_rf = Pipeline(stages=[assembler, rf])

# Cross validation
# The data is split into folds for training and testing, the best performing model is selected
# numFolds=5 so the dataset will be split into 5 parts, train the data on 4 subsets and test in om the remaning subset.Repeat this process for each subset. 
# Choose the hyperparametere that on average, lead to the best model performance
crossval_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=parameterGrid_lr,evaluator=BinaryClassificationEvaluator(),numFolds=5)

crossval_rf = CrossValidator(estimator=pipeline_rf,estimatorParamMaps=parameterGrid_rf,evaluator=BinaryClassificationEvaluator(),numFolds=5)

# Fit the models
# fit () is used to train the model - training data
cvModel_lr = crossval_lr.fit(train_data)
cvModel_rf = crossval_rf.fit(train_data)

# Predict on validation data
# The transform() function takes the input data, applies the model to it, and outputs a dataframe that includes a column of predicted labels.
prediction_lr_val = cvModel_lr.transform(validation_data)
prediction_rf_val = cvModel_rf.transform(validation_data)

# Evaluate the models on validation data
# evaluator takes in the predictiion result of the binary classification of the model and computes the performace metrics based oni the results.
evaluator = BinaryClassificationEvaluator()
print("Validation Area Under ROC Logistic Regression: " + str(evaluator.evaluate(prediction_lr_val, {evaluator.metricName: "areaUnderROC"})))
print("Validation Area Under ROC Random Forest: " + str(evaluator.evaluate(prediction_rf_val, {evaluator.metricName: "areaUnderROC"})))

# Predict on test data
prediction_lr_test = cvModel_lr.transform(test_data)
prediction_rf_test = cvModel_rf.transform(test_data)

# Evaluate the models on test data
print("Test Area Under ROC Logistic Regression: " + str(evaluator.evaluate(prediction_lr_test, {evaluator.metricName: "areaUnderROC"})))
print("Test Area Under ROC Random Forest: " + str(evaluator.evaluate(prediction_rf_test, {evaluator.metricName: "areaUnderROC"})))

# Calculate Precision, recall and F1 score
# MulticlassMetrics takes an input RDD of prediction,label - pairs and computes the metrics 
# MulticlassMetrics also provides acces to metrics Preccison,recall and FMeasure-F1
# lambda function to create new rdd which will only include label column, distinct to remove duplicates and then collect the data
# Precision the number of true positive divided by the number of elements labeled belonging to the class
# Recall the number of tru positives divited by the total number of element that actually belong to the class
# F1 scorethe harmonic mean of the precission and recall. F1 = 2 *(Precision*Recall)/(Precision+Recall)

def print_metrics(predictions, model_name):
    print("\nMetrics for : %s" % model_name)
    predictions_and_labels = predictions.select(['prediction','label']).rdd
    metrics = MulticlassMetrics(predictions_and_labels)
    labels = predictions_and_labels.map(lambda lp: lp.label).distinct().collect()
    for label in sorted(labels):
        original_label = 'Malignant' if label else 'Benign'
        print("Class %s precision = %s" % (original_label, metrics.precision(label)))
        print("Class %s recall = %s" % (original_label, metrics.recall(label)))
        print("Class %s F1 Score = %s" % (original_label, metrics.fMeasure(label, beta=1.0)))
    print("\n")

print_metrics(prediction_lr_test, "Logistic Regression")
print_metrics(prediction_rf_test, "Random Forest")


