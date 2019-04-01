import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType

object Survival {
  def main(args: Array[String]) {

    val spark = SparkSession
      .builder
      .appName("Titanic Survival")
      .master("local") // remove this when running in a Spark cluster
      .getOrCreate()

    println("Connected to Spark")

    // Display only ERROR logs in terminal
    spark.sparkContext.setLogLevel("ERROR")

    // Specify data file
    val filePath = "data/titanic.csv"

    val passengers = spark.read.option("header", "true").option("inferSchema", "true").csv(filePath)

    // How many passengers were there on titanic?
    println("There were total " + passengers.count() + " passengers")

    // Get the columns that we need further
    val passengers1 = passengers.select(
      passengers("Pclass"),
      passengers("Survived").cast(DoubleType).as("Survived"),
      passengers("Gender"),
      passengers("Age"),
      passengers("SibSp"),
      passengers("Parch"),
      passengers("Fare")
    )

    // print the schema of passengers1
    passengers1.printSchema()

    // Find the gender distribution of passengers
    val gender_dist = passengers1.groupBy("Gender").count().as("Gender")
    println("Total")
    gender_dist.show()

    // Find the number of males and females that survived
    import spark.implicits._
    val survived_gender_dist = passengers1.filter($"Survived" === 1.0).groupBy("Gender").count()
    println("Survived")
    survived_gender_dist.show()

    // VectorAssembler does not support the StringType type. So convert Gender to numeric
    val indexer = new StringIndexer()
      .setInputCol("Gender")
      .setOutputCol("GenderCat")

    val passengers2 = indexer.fit(passengers1).transform(passengers1)

    //println("Drop instances with empty values")
    val passengers3 = passengers2.na.drop()
    //println("Orig = "+passengers2.count()+" Final = "+ passengers3.count() + "Dropped = "+ (passengers2.count() - passengers3.count()))

    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass", "GenderCat", "Age", "SibSp", "Parch", "Fare"))
      .setOutputCol("features")

    val passengers4 = assembler.transform(passengers3)

    val Array(train, test) = passengers4.randomSplit(Array(0.9, 0.1))
    //println("Train = "+train.count()+" Test = "+test.count())

    val classifier = new DecisionTreeClassifier()
      .setLabelCol("Survived")
      .setImpurity("entropy") // could be "gini"
      .setMaxBins(32)
      .setMaxDepth(5)

    val model = classifier.fit(train)

    println("The Decision tree has %d nodes.".format(model.numNodes))
    println(model.toDebugString)
    println(model.toString)
    println(model.featureImportances)

    val predictions = model.transform(test)
    predictions.show(10)

    // model evaluation
    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("Survived")
    evaluator.setMetricName("weightedRecall") // could be f1, "weightedPrecision" or "weightedRecall"

    val startTime = System.nanoTime()
    val recall = evaluator.evaluate(predictions)
    println("Test Recall = %.2f%%".format(recall * 100))

    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println("Elapsed time: %.2fseconds".format(elapsedTime))

    spark.stop()
    println("Disconnected from Spark")

  }

}
