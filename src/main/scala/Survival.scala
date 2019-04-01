import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature._
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
    val filePath = "titanic.csv"

    val passengers = spark.read.option("header","true"). option("inferSchema","true"). csv(filePath)
    val passengers1 = passengers
      .select(passengers("Pclass"),
        passengers("Survived").cast(DoubleType).as("Survived"),
        passengers("Gender"),
        passengers("Age"),
        passengers("SibSp"),
        passengers("Parch"),
        passengers("Fare")
      )

    //
    // VectorAssembler does not support the StringType type. So convert Gender to numeric
    //
    val indexer = new StringIndexer()
      .setInputCol("Gender")
      .setOutputCol("GenderCat")

    val passengers2 = indexer.fit(passengers1).transform(passengers1)

    val passengers3 = passengers2.na.drop()

    println("Orig = "+passengers2.count()+" Final = "+ passengers3.count() + "Dropped = "+ (passengers2.count() - passengers3.count()))

    val assembler = new VectorAssembler()
      .setInputCols(Array("Pclass","GenderCat","Age","SibSp","Parch","Fare"))
      .setOutputCol("features")

    val passengers4 = assembler.transform(passengers3)

    val Array(train, test) = passengers4.randomSplit(Array(0.9, 0.1))
    println("Train = "+train.count()+" Test = "+test.count())

    val algTree = new DecisionTreeClassifier()
      .setLabelCol("Survived")
      .setImpurity("entropy") // could be "gini"
      .setMaxBins(32)
      .setMaxDepth(5)

    val mdlTree = algTree.fit(train)
    println("The tree has %d nodes.".format(mdlTree.numNodes))
    println(mdlTree.toDebugString)
    println(mdlTree.toString)
    println(mdlTree.featureImportances)

    val predictions = mdlTree.transform(test)
    predictions.show(5)

    // model evaluation
    val evaluator = new MulticlassClassificationEvaluator()
    evaluator.setLabelCol("Survived")
    evaluator.setMetricName("accuracy") // could be f1, "weightedPrecision" or "weightedRecall"
    //
    val startTime = System.nanoTime()
    val accuracy = evaluator.evaluate(predictions)
    println("Test Accuracy = %.2f%%".format(accuracy*100))
    //
    val elapsedTime = (System.nanoTime() - startTime) / 1e9
    println("Elapsed time: %.2fseconds".format(elapsedTime))


    spark.stop()
    println("Disconnected from Spark")

  }

}
