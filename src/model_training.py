from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

def build_model(df):
    # Features to vector
    feature_cols = df.columns
    feature_cols.remove("Class")

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Random Forest Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="Class", numTrees=100)

    pipeline = Pipeline(stages=[assembler, rf])

    model = pipeline.fit(df)
    return model
