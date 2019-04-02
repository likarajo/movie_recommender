/************************************************************
  * This class requires two arguments:
  *  input file
  *  output location - can be on S3 or cluster
  *************************************************************/

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

// Declare record structure as a class
case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
case class Movie(mId: Int, movieName: String, genre: String)

object BookRecommender {
  def main(args: Array[String]): Unit = {

    if (args.length == 0) {
      println("i need two two parameters ")
    }

    val spark = SparkSession
      .builder()
      .appName("BookReco1")
      .getOrCreate()


    val ratings = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .csv(args(0))

    val indexer = new StringIndexer().setInputCol("ISBN").setOutputCol("ISBNIndex")
    val indexed = indexer.fit(ratings).transform(ratings)
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

    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)

  }
}
