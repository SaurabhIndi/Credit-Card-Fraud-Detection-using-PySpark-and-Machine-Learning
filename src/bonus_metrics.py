from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(model, df, output_path="confusion_matrix.png"):
    # Generate predictions
    predictions = model.transform(df)
    
    # Select true and predicted labels
    preds_and_labels = predictions.select("Class", "prediction").toPandas()
    
    # Calculate confusion matrix using sklearn
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(preds_and_labels["Class"], preds_and_labels["prediction"])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def plot_feature_importance(model, feature_cols, output_path="feature_importance.png"):
    # Extract RandomForest model from pipeline
    rf_model = model.stages[-1]
    
    # Get feature importances
    importances = rf_model.featureImportances.toArray()
    
    # Create DataFrame for plotting
    feature_importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
    plt.title("Feature Importance from Random Forest")
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")