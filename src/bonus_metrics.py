from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(model, df, output_path="confusion_matrix.png"):
    predictions = model.transform(df)
    preds_and_labels = predictions.select("Class", "prediction").toPandas()
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(preds_and_labels["Class"], preds_and_labels["prediction"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_feature_importance(model, feature_cols, output_path="feature_importance.png"):
    rf_model = model.bestModel.stages[-1]
    importances = rf_model.featureImportances.toArray()
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance from Random Forest")
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")