/************************************************************
  * This class requires two arguments:
  *  input file
  *  output location - can be on S3 or cluster
  *************************************************************/

import org.apache.spark.sql.SparkSession

// Declare record structure as a class
case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
case class Movie(movieId: Int, movieName: String, genre: String)

object Recommender {

  def main(args: Array[String]) {

    //if (args.length == 0) {println("i need two two parameters ")}

    val spark = SparkSession
      .builder
      .appName("Movie Recommender")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

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

    ratings.show()

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

    movies.show()

    /*

    println("Data read completed")

    // Split data into training and testing set
    val Array(train, test) = ratings.randomSplit(Array(0.9, 0.1))

    // Set Algorithm
    val als = new ALS()
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
      .setMaxIter(5)
      .setRegParam(0.01)
      .setRank(5)

    println("Training model with train set...")

    //Create Model with train data
    val model = als.fit(train)

    // Drop any rows in the DataFrame of predictions that contain NaN values
    model.setColdStartStrategy("drop")

    println("Model trained")

    println("Predicting using test set...")

    // Predict using test data
    val predictions = model.transform(test)

    predictions.show()

    // Model evaluation
    val evaluator = new RegressionEvaluator()
      .setLabelCol("rating")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    println("Evaluating model...")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // Generate top 5 movie recommendations for each user
    println("Generating Top 5 movie recommendations for each user...")
    val userRecs = model.recommendForAllUsers(5)
    userRecs.rdd.foreach(println)
    userRecs.rdd.saveAsTextFile("output")

    /*
    // Generate top 5 user recommendations for each movie
    println("Generating Top 5 user recommendations for each movie...")
    val movieRecs = model.recommendForAllItems(5)
    movieRecs.rdd.foreach(println)

    // Generate top 5 movie recommendations for a specified set of users
    println("Generating Top 5 movie recommendations for a specified set of users")
    userRecs.rdd.repartition(1).saveAsTextFile("output")
    //userRecs.rdd.repartition(1).foreach(println)
    println("Saved to output")
    */

    spark.stop()
    println("Disconnected from Spark")
    */

  }

}
