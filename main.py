from src.preprocessing import create_spark_session, load_data, preprocess_data
from src.model_training import build_model
from src.evaluation import evaluate_model
from src.bonus_metrics import plot_confusion_matrix, plot_feature_importance

def main():
    spark = create_spark_session()
    
    # Load and preprocess data
    df = load_data(spark, "data/creditcard.csv")
    df = preprocess_data(df)
    
    # Split data into train and test (optional for better evaluation)
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Train model
    model = build_model(train_df)

    # Evaluate model
    evaluate_model(model, test_df)

    # Bonus: Plot confusion matrix and feature importance
    feature_cols = [col for col in df.columns if col != "Class"]
    plot_confusion_matrix(model, test_df)
    plot_feature_importance(model, feature_cols)

    spark.stop()

if __name__ == "__main__":
    main()