import pytest
from src.preprocessing import preprocess_data
from pyspark.sql import SparkSession

@pytest.fixture
def spark():
    spark = SparkSession.builder.appName("Test").getOrCreate()
    yield spark
    spark.stop()

def test_preprocess_data(spark):
    # Create a small test DataFrame
    data = [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1)]
    columns = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"]
    df = spark.createDataFrame(data, columns)
    
    # Preprocess the DataFrame
    processed_df = preprocess_data(df)
    
    # Assert no nulls and correct type for Class
    assert processed_df.count() == 1
    assert processed_df.schema["Class"].dataType.simpleString() == "int"