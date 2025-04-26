from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType

def evaluate_model(model, df):
    predictions = model.transform(df)

    # Binary evaluator for AUC
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    auc = evaluator_auc.evaluate(predictions)
    print(f"Test AUC: {auc:.4f}")

    # Multiclass evaluator for precision, recall, F1
    evaluator_multi = MulticlassClassificationEvaluator(
        labelCol="Class", predictionCol="prediction", metricName="weightedPrecision"
    )
    precision = evaluator_multi.evaluate(predictions)
    recall = MulticlassClassificationEvaluator(
        labelCol="Class", predictionCol="prediction", metricName="weightedRecall"
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol="Class", predictionCol="prediction", metricName="f1"
    ).evaluate(predictions)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Extract probability of class 1 for threshold adjustment
    def adjust_threshold(prob_array):
        return float(prob_array[1]) * 0.3  # Use probability of class 1 and adjust threshold

    udf_adjust = udf(adjust_threshold, FloatType())
    thresholded_preds = predictions.withColumn("rawPredictionAdjusted", 
                                             udf_adjust(col("probability")))
    thresholded_preds = thresholded_preds.withColumn("predictionAdjusted", 
                                                    (col("rawPredictionAdjusted") > 0.5).cast("integer"))
    
    auc_adjusted = evaluator_auc.evaluate(thresholded_preds)
    print(f"Adjusted AUC (threshold 0.3): {auc_adjusted:.4f}")

    # Log cross-validation metrics if available
    if hasattr(model, "avgMetrics"):
        print(f"Cross-Validation Average AUC: {sum(model.avgMetrics) / len(model.avgMetrics):.4f}")