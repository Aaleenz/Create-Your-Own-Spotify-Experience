from pyspark.sql import SparkSession
from pyspark.sql.functions import col, substring_index, explode
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Creating a SparkSession
spark = SparkSession.builder \
    .appName("MusicRecommendation") \
    .config("spark.mongodb.input.uri", "mongodb://localhost:27017/music_database.music_collection") \
    .config("spark.mongodb.output.uri", "mongodb://localhost:27017/music_database.music_collection") \
    .getOrCreate()

# Loading data from MongoDB into a Spark DataFrame
df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()

# Extracting 'track_id' from 'audio_file'
df = df.withColumn('track_id', substring_index(col('audio_file'), '.', 1).substr(2, 100).cast('integer'))

# Reading CSV data containing 'track_favorites'
csv_data = spark.read.csv('/home/hadoop/Downloads/fma_metadata/raw_tracks.csv', header=True,
                          inferSchema=True, nullValue='')
csv_data = csv_data.withColumnRenamed("track_id", "csv_track_id")  # Renaming the column to avoid ambiguity
csv_data = csv_data.withColumn("track_favorites", csv_data["track_favorites"].cast("int"))

# Replace NaN values in 'track_favorites' with a default value (e.g., 0)
csv_data = csv_data.fillna({'track_favorites': 0})

# Selecting required columns from CSV
csv_data = csv_data.select('csv_track_id', 'track_favorites')

# Prepare DataFrame for LSH model training
lsh_df = df.join(csv_data, df['track_id'] == csv_data['csv_track_id'], 'inner')

# Flatten the 'features' column
lsh_df = lsh_df.withColumn('features', explode(col('features')))

# Define a function to convert array to vector
to_vector = udf(lambda a: Vectors.dense(a.toArray()), VectorUDT())

# Convert 'features' column to vector
lsh_df = lsh_df.withColumn('features_vector', to_vector(col('features')))

# Optionally, dropping the 'features' column if it's no longer needed
lsh_df = lsh_df.drop('features')

# Define MinHashLSH model
mh = MinHashLSH(inputCol="features_vector", outputCol="hashes", numHashTables=5)

# Fit the LSH model
model = mh.fit(lsh_df)

# Transform the data
hashed_df = model.transform(lsh_df)

# Define the evaluator
evaluator = RegressionEvaluator(metricName="rmse", labelCol="track_favorites", predictionCol="prediction")

# Define the parameter grid for hyperparameter tuning
paramGrid = ParamGridBuilder() \
    .addGrid(mh.numHashTables, [3, 5, 10]) \
    .build()

# Define cross-validation
crossval = CrossValidator(estimator=mh,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)  # Use 3 folds for cross-validation

# Split the data into training and testing sets
(trainingData, testData) = hashed_df.randomSplit([0.8, 0.2], seed=1234)

# Fit the model
cvModel = crossval.fit(trainingData)

# Make predictions on the test data
predictions = cvModel.transform(testData)

# Evaluate the model on test data
rmse = evaluator.evaluate(predictions)

print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# Best model
best_model = cvModel.bestModel

# Print best model parameters
print("Best number of Hash Tables:", best_model._java_obj.getNumHashTables())

