from src.preprocessing import create_spark_session, load_data, preprocess_data
from src.model_training import build_model
from src.evaluation import evaluate_model

def main():
    spark = create_spark_session()
    
    # Load and preprocess data
    df = load_data(spark, "data/creditcard.csv")
    df = preprocess_data(df)

    # Train model
    model = build_model(df)

    # Evaluate model
    evaluate_model(model, df)

    spark.stop()

if __name__ == "__main__":
    main()
