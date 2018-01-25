import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.rdd._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import scala.collection.mutable.ListBuffer
import org.apache.log4j.Logger
import imputationClustering.parserMV
import imputationClustering.imputationKmeans

object runImpKmeans extends Serializable {

  var sc: SparkContext = null

  def main(arg: Array[String]) {
    var logger = Logger.getLogger(this.getClass())
    var k_neighbour, numMapsKnn, numReducesKnn, numIterations = 0
    //Reading parameters
    val pathInput = arg(0)
    val pathHeader = arg(1)
    val pathOutput = arg(2)
    val K = arg(3).toInt
    val maxInterations = arg(4).toInt
    val seed = arg(5).toInt
    val epsilon = arg(6).toDouble
    val numPartitionMap = arg(7).toInt
    val imputationType = arg(8)
    if (imputationType == "knn"){
      k_neighbour = arg(9).toInt
      numMapsKnn = arg(10).toInt
      numReducesKnn = arg(11).toInt
      numIterations = arg(12).toInt
    }

    val jobName = "Imputation - K-Means -> " + " K = " + K

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    sc = new SparkContext(conf)
    
    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathInput \"" + pathInput + "\"")
    logger.info("=> pathHeader \"" + pathHeader + "\"")
    logger.info("=> pathOutput \"" + pathOutput + "\"")
    logger.info("=> NumberClusters \"" + K + "\"")
    logger.info("=> MaxIterations \"" + maxInterations + "\"")
    logger.info("=> Seed \"" + seed + "\"")
    logger.info("=> Epsilon \"" + epsilon + "\"")
    logger.info("=> NumberMapPartition \"" + numPartitionMap + "\"")
    logger.info("=> Type of imputation \"" + imputationType + "\"")
    if (imputationType == "knn") {
      logger.info("=> NumberNeighbors \"" + k_neighbour + "\"")
      logger.info("=> NumberMapPartition \"" + numMapsKnn + "\"")
      logger.info("=> NumberReducePartition \"" + numReducesKnn + "\"")
      logger.info("=> Number of iterations \"" + numIterations + "\"")
    }
    
    // Load data and header
    val parserMV = new parserMV(sc, pathHeader)
    val data = sc.textFile(pathInput, numPartitionMap).map(parserMV.parserToVector(_)).persist()
    
    // Imputation
    val km = new imputationKmeans(K, maxInterations, seed, epsilon, parserMV, pathOutput)
    if (imputationType == "centroid") {
      km.imputation(data)
    }else if (imputationType == "knn") {
      km.imputationKNN(data, k_neighbour, numMapsKnn, numReducesKnn, numIterations)
    }
    
    logger.info("=> Time of Imputation \"" + km.getTime + "\"")
    
  }

}