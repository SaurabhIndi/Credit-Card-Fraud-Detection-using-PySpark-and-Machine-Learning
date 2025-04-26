from src.preprocessing import create_spark_session, load_data, preprocess_data
from src.model_training import build_model, apply_smote
from src.evaluation import evaluate_model

def main():
    spark = create_spark_session()
    
    # Load and preprocess data
    df = load_data(spark, "data/creditcard.csv")
    df = preprocess_data(df)
    
    # Split data into train and test
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    # Apply SMOTE to handle class imbalance
    train_df_balanced = apply_smote(train_df)

    # Train model with cross-validation
    model = build_model(train_df_balanced)

    # Evaluate model
    evaluate_model(model, test_df)

    spark.stop()

if __name__ == "__main__":
    main()