/************************************************************
  * This class requires two arguments:
  *  input file
  *  output location - can be on S3 or cluster
  *************************************************************/

import java.io.{BufferedWriter, ByteArrayOutputStream, File, FileWriter}
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

// Declare record structure as a class
case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
case class Movie(mId: Int, movieName: String, genre: String)

object Recommender {

  def main(args: Array[String]) {

    //if (args.length == 0) {println("i need two two parameters ")}

    val debug = false
    val file = true

    val spark = SparkSession
      .builder
      .appName("Movie Recommender")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark. Running...")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

    // Get current time
    val xt = LocalDateTime.now.format(DateTimeFormatter.ofPattern("YYMMddHHmmss"))

    // Specify output file
    val filename = "MovieRecommendation_" + xt + ".txt"
    val outFile = new BufferedWriter(new FileWriter(filename))

    // Specify data file(s)
    val ratingsFile = "movielens/ratings.dat"
    val moviesFile = "movielens/movies.dat"

    import spark.implicits._

    // Create the ratings dataframe using the Rating data structure

    def parseRating(str: String): Rating = {
      val fields = str.split("::")
      assert(fields.size == 4)
      Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
    }

    val ratings = spark.read.option("header", "false").option("inferSchema", "true")
      .textFile(ratingsFile)
      .map(parseRating)
      .toDF()

    //ratings.show()

    // Create the movies dataframe using the Movie data structure

    def parseMovie(str: String): Movie = {
      val fields = str.split("::")
      assert(fields.size == 3)
      Movie(fields(0).toInt, fields(1), fields(2))
    }

    val movies = spark.read.option("header", "false").option("inferSchema", "true")
      .textFile(moviesFile)
      .map(parseMovie)
      .toDF()

    //movies.show()

    if(debug) println("Data read completed")
    if(file) outFile.append("Data read completed\n")

    val df = ratings.join(movies, $"movieId" === $"mId")
      .select("userId", "movieId", "movieName", "genre", "rating", "timestamp")

    if(debug) df.show()
    if(file){
      val outCapture = new ByteArrayOutputStream()
      Console.withOut(outCapture) {df.show()}
      val result = new String(outCapture.toByteArray)
      outFile.append(result)
    }


    // Split data into training and testing set
    val Array(train, test) = df.randomSplit(Array(0.9, 0.1))

    // Set Algorithm
    val als = new ALS()
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
      .setMaxIter(5)
      .setRegParam(0.01)
      .setRank(5)

    if(debug) println("Training model with train set...")
    if(file) outFile.append("Training model with train set...\n")

    //Create Model with train data
    val model = als.fit(train)

    // Drop any rows in the DataFrame of predictions that contain NaN values
    model.setColdStartStrategy("drop")

    if(debug) println("Model trained")
    if(file) outFile.append("Model trained\n")

    if(debug) println("Predicting using test set...")
    if(file) outFile.append("Predicting using test set...\n")

    // Predict using test data
    val predictions = model.transform(test)

    if(debug) predictions.show()
    if(file){
      val outCapture = new ByteArrayOutputStream()
      Console.withOut(outCapture) {predictions.show()}
      val result = new String(outCapture.toByteArray)
      outFile.append(result)
    }

    // Model evaluation
    val evaluator = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    if(debug) println("Evaluating model...")
    if(file) outFile.append("Evaluating model...\n")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error of the ALS model = $rmse")
    if(file) outFile.append(s"Root-mean-square error of the ALS model = $rmse\n")

    // Generate top 5 movie recommendations for each user
    println("Generating Top 5 movie recommendations for each user...")
    if(file) outFile.append("Generating Top 5 movie recommendations for each user...\n")
    val userRecs = model.recommendForAllUsers(5)

    userRecs.toDF().show()
    if(file){
      val outCapture = new ByteArrayOutputStream()
      Console.withOut(outCapture) {userRecs.toDF().show()}
      val result = new String(outCapture.toByteArray)
      outFile.append(result)
    }

    /*
    // Generate top 5 user recommendations for each movie
    println("Generating Top 5 user recommendations for each movie...")
    val movieRecs = model.recommendForAllItems(5)

    // Generate top 5 movie recommendations for a specified set of users
    println("Generating Top 5 movie recommendations for a specified set of users")
    val top5recs = userRecs.rdd.repartition(1)
    */

    spark.stop()
    println("Disconnected from Spark")

    outFile.close()
    if(!file) new File(filename).delete()

  }

}
