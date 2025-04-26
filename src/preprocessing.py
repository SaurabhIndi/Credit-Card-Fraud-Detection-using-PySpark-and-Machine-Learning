from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Credit Card Fraud Detection") \
        .getOrCreate()
    return spark

def load_data(spark, path):
    df = spark.read.csv(path, header=True, inferSchema=True)
    return df

def preprocess_data(df):
    # Drop any null values
    df = df.dropna()
    # Optional: cast "Class" column to Integer
    df = df.withColumn("Class", col("Class").cast("integer"))
    return df
