from pyspark.ml.evaluation import BinaryClassificationEvaluator

def evaluate_model(model, df):
    predictions = model.transform(df)

    evaluator = BinaryClassificationEvaluator(
        labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    
    auc = evaluator.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")


def plot_confusion_matrix(predictions):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    y_true = predictions.select("Class").collect()
    y_pred = predictions.select("prediction").collect()
    
    y_true = [int(row['Class']) for row in y_true]
    y_pred = [int(row['prediction']) for row in y_pred]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
