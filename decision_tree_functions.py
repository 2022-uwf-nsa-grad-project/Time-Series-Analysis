import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from sklearn.metrics import confusion_matrix
from openpyxl import Workbook

def train_and_evaluate(train_df, test_df, tactic, train_year, test_year, split_ratio):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    feature_columns = ["duration", "orig_bytes", "resp_bytes", "orig_ip_bytes", "resp_ip_bytes"]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
    train_df = assembler.transform(train_df)
    test_df = assembler.transform(test_df)

    train_df = train_df.withColumn("label", F.when(F.col("label_tactic") == tactic, 1).otherwise(0))
    test_df = test_df.withColumn("label", F.when(F.col("label_tactic") == tactic, 1).otherwise(0))

    dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    model = dt.fit(train_df)

    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)

    precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    precision = precision_evaluator.evaluate(predictions)
    recall = recall_evaluator.evaluate(predictions)
    f1_score = f1_evaluator.evaluate(predictions)

    print(f"\nModel for label_tactic: {tactic}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

    # Print predictions for each row
    print("\nPredictions for each row in the test dataframe:")
    predictions.select("features", "label", "prediction").show(truncate=False)

    # Create a workbook and add a worksheet
    excel_path = "Model_Results.xlsx"
    if os.path.exists(excel_path):
        os.remove(excel_path)
    wb = Workbook()
    ws = wb.active
    ws.title = "Model Results"
    ws.append(["Tactic", "Accuracy", "Precision", "Recall", "F1 Score", "Train Year", "Test Year", "Split Ratio"])

    # Write results to Excel
    ws.append([tactic, accuracy, precision, recall, f1_score, train_year, test_year, split_ratio])

    # Save the workbook
    wb.save(excel_path)

    # Confusion Matrix for all predictions
    y_true = [row['label'] for row in predictions.select('label').collect()]
    y_pred = [row['prediction'] for row in predictions.select('prediction').collect()]

    # Print lengths of y_true and y_pred for debugging
    print(f"Length of y_true: {len(y_true)}")
    print(f"Length of y_pred: {len(y_pred)}")

    # Check for missing values in y_true and y_pred
    print(f"Number of missing values in y_true: {sum(pd.isnull(y_true))}")
    print(f"Number of missing values in y_pred: {sum(pd.isnull(y_pred))}")

    # Ensure lengths match by filtering out rows without predictions
    if len(y_true) != len(y_pred):
        min_length = min(len(y_true), len(y_pred))
        y_true = y_true[:min_length]
        y_pred = y_pred[:min_length]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {tactic}')
    plt.show()
    print("Confusion Matrix plotted.")

# Function to split data into training and testing sets
def split_data(df, tactic, train_ratio):
    attack_df = df.filter(F.col("label_tactic") == tactic)
    benign_df = df.filter(F.col("label_tactic") == "none")
    attack_train, attack_test = attack_df.randomSplit([train_ratio, 1 - train_ratio], seed=42)
    benign_train, benign_test = benign_df.randomSplit([train_ratio, 1 - train_ratio], seed=42)
    train_df = attack_train.union(benign_train)
    test_df = attack_test.union(benign_test)
    return train_df, test_df

#Function to train and test the model on the same year, 2022
def train_test_2022_2022(combined_df_2022, tactic, train_ratio, split_ratio):
    print(f"\nTrain 2022 data, Test 2022 data ({split_ratio})")
    train_df, test_df = split_data(combined_df_2022, tactic, train_ratio)
    print(f"Arguments: combined_df_2022, tactic={tactic}, split_ratio={split_ratio}")
    train_and_evaluate(train_df, test_df, tactic, 2022, 2022, split_ratio)

# Function to train and test the model on the same year, 2024
def train_test_2024_2024(combined_df_2024, tactic, train_ratio, split_ratio):
    print(f"\nTrain 2024 data, Test 2024 data ({split_ratio})")
    train_df, test_df = split_data(combined_df_2024, tactic, train_ratio)
    train_and_evaluate(train_df, test_df, tactic, 2024, 2024, split_ratio)

# Function to train the model on 2022 data and test on 2024 data
def train_2022_test_2024(combined_df_2022, combined_df_2024, tactic, train_ratio, test_ratio, split_ratio):
    print(f"\nTrain 2022 data, Test 2024 data ({split_ratio})")
    train_df, _ = split_data(combined_df_2022, tactic, train_ratio)
    _, test_df = split_data(combined_df_2024, tactic, test_ratio)
    train_and_evaluate(train_df, test_df, tactic, 2022, 2024, split_ratio)

# Function to train the model on 2024 data and test on 2022 data
def train_2024_test_2022(combined_df_2024, combined_df_2022, tactic, train_ratio, test_ratio, split_ratio):
    print(f"\nTrain 2024 data, Test 2022 data ({split_ratio})")
    train_df, _ = split_data(combined_df_2024, tactic, train_ratio)
    _, test_df = split_data(combined_df_2022, tactic, test_ratio)
    train_and_evaluate(train_df, test_df, tactic, 2024, 2022, split_ratio)

# Functions for training and testing with oversampled training data

#Function to train and test the model on the same year, 2022
def train_test_2022_2022_oversampled(combined_df_2022, tactic, train_ratio, split_ratio):
    print(f"\nTrain 2022 data, Test 2022 data ({split_ratio})")
    train_df, test_df = split_data(combined_df_2022, tactic, train_ratio)
    train_df = duplicate_and_shuffle(train_df)
    print(f"Arguments: combined_df_2022, tactic={tactic}, split_ratio={split_ratio}")
    train_and_evaluate(train_df, test_df, tactic, 2022, 2022, split_ratio)

# Function to train and test the model on the same year, 2024
def train_test_2024_2024_oversampled(combined_df_2024, tactic, train_ratio, split_ratio):
    print(f"\nTrain 2024 data, Test 2024 data ({split_ratio})")
    train_df, test_df = split_data(combined_df_2024, tactic, train_ratio)
    train_df = duplicate_and_shuffle(train_df)
    train_and_evaluate(train_df, test_df, tactic, 2024, 2024, split_ratio)

# Function to train the model on 2022 data and test on 2024 data
def train_2022_test_2024_oversampled(combined_df_2022, combined_df_2024, tactic, train_ratio, test_ratio, split_ratio):
    print(f"\nTrain 2022 data, Test 2024 data ({split_ratio})")
    train_df, _ = split_data(combined_df_2022, tactic, train_ratio)
    _, test_df = split_data(combined_df_2024, tactic, test_ratio)
    train_df = duplicate_and_shuffle(train_df)
    train_and_evaluate(train_df, test_df, tactic, 2022, 2024, split_ratio)

# Function to train the model on 2024 data and test on 2022 data
def train_2024_test_2022_oversampled(combined_df_2024, combined_df_2022, tactic, train_ratio, test_ratio, split_ratio):
    print(f"\nTrain 2024 data, Test 2022 data ({split_ratio})")
    train_df, _ = split_data(combined_df_2024, tactic, train_ratio)
    _, test_df = split_data(combined_df_2022, tactic, test_ratio)
    train_df = duplicate_and_shuffle(train_df)
    train_and_evaluate(train_df, test_df, tactic, 2024, 2022, split_ratio)

# Function to duplicate, concatenate, and shuffle the dataframe
def duplicate_and_shuffle(df):
    # Duplicate the dataframe
    duplicated_df = df.union(df)
    # Shuffle the dataframe
    shuffled_df = duplicated_df.orderBy(F.rand())
    return shuffled_df

