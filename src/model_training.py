from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import explode, lit, col, array, rand

def apply_smote(df):
    # Filter majority and minority classes
    majority_df = df.filter(col("Class") == 0).cache()
    minority_df = df.filter(col("Class") == 1).cache()
    
    # Calculate counts
    majority_count = majority_df.count()
    minority_count = minority_df.count()
    
    # Handle edge case: if minority count is 0, return original DataFrame
    if minority_count == 0:
        majority_df.unpersist()
        minority_df.unpersist()
        return df
    
    # Calculate oversampling ratio (ensure at least 1 to avoid zero division)
    ratio = max(1, int(majority_count / minority_count))
    
    # Oversample minority class
    oversampled_minority = minority_df.withColumn("dummy", explode(array([lit(i) for i in range(ratio)]))).drop("dummy")
    balanced_df = majority_df.union(oversampled_minority)
    
    # Shuffle to randomize
    balanced_df = balanced_df.orderBy(rand())
    
    # Cleanup: unpersist cached DataFrames
    majority_df.unpersist()
    minority_df.unpersist()
    
    return balanced_df

def build_model(df):
    # Features to vector
    feature_cols = [col for col in df.columns if col != "Class"]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Random Forest Classifier
    rf = RandomForestClassifier(featuresCol="features", labelCol="Class", numTrees=100)

    pipeline = Pipeline(stages=[assembler, rf])

    # Parameter grid for cross-validation
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [100, 200]) \
        .addGrid(rf.maxDepth, [10, 20]) \
        .build()

    # Cross-validator
    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=BinaryClassificationEvaluator(labelCol="Class"),
                              numFolds=3)

    # Fit the model
    model = crossval.fit(df)
    return model