/************************************************************
  * This class requires two arguments:
  *  input file
  *  output location - can be on S3 or cluster
  *************************************************************/

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

object BookRecommender {

  def main(args: Array[String]){

    //if (args.length == 0) {println("i need two two parameters ")}

    val spark = SparkSession
      .builder
      .appName("Book Recommender")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark. Running...")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

    val ratingsFile = "data/informatik/BX-Book-Ratings.csv"

    val ratings = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .csv(ratingsFile)

    ratings.show()

    val indexer = new StringIndexer()
      .setInputCol("ISBN")
      .setOutputCol("ISBNIndex")

    val indexed = indexer
      .fit(ratings)
      .transform(ratings)

    val Array(training, test) = indexed.randomSplit(Array(0.8, 0.2))

    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setRank(10)
      .setUserCol("User-ID")
      .setItemCol("ISBNIndex")
      .setRatingCol("Book-Rating")

    val model = als.fit(training)
    model.setColdStartStrategy("drop")

    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("Book-Rating")
      .setPredictionCol("prediction")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // Generate top 10 nook recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    userRecs.toDF.show()

    // Generate top 10 user recommendations for each book
    val bookRecs = model.recommendForAllItems(10)
    bookRecs.toDF.show()

    spark.stop()
    println("Disconnected from Spark")

  }
}
