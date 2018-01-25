

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
import imputationClustering.imputationFuzzyKmeans
import imputationClustering.imputationKmeans
import imputationClustering.imputationFuzzyKmeans

object runImpFuzzyKmeans extends Serializable {

  var sc: SparkContext = null

  def main(arg: Array[String]) {
    var logger = Logger.getLogger(this.getClass())
    var k_neighbour, numMapsKnn, numReducesKnn, numIterations = 0
    //Reading parameters
    val pathInput = arg(0)
    val pathHeader = arg(1)
    val pathOutput = arg(2)
    val K = arg(3).toInt
    val M = arg(4).toDouble
    val maxInterations = arg(5).toInt
    val seed = arg(6).toInt
    val epsilon = arg(7).toDouble
    val numPartitionMap = arg(8).toInt
   

    val jobName = "Imputation - Fuzzy-K-Means -> " + " K = " + K

    //Spark Configuration
    val conf = new SparkConf().setAppName(jobName)
    sc = new SparkContext(conf)
    
    logger.info("=> jobName \"" + jobName + "\"")
    logger.info("=> pathInput \"" + pathInput + "\"")
    logger.info("=> pathHeader \"" + pathHeader + "\"")
    logger.info("=> pathOutput \"" + pathOutput + "\"")
    logger.info("=> NumberClusters \"" + K + "\"")
    logger.info("=> Fuzzifier \"" + M + "\"")
    logger.info("=> MaxIterations \"" + maxInterations + "\"")
    logger.info("=> Seed \"" + seed + "\"")
    logger.info("=> Epsilon \"" + epsilon + "\"")
    logger.info("=> NumberMapPartition \"" + numPartitionMap + "\"")
    
    // Load data and header
    val parserMV = new parserMV(sc, pathHeader)
    val data = sc.textFile(pathInput, numPartitionMap).map(parserMV.parserToVector(_)).persist()
    
    // Imputation
    val fkm = new imputationFuzzyKmeans(K, M, maxInterations, seed, epsilon, parserMV, pathOutput)
    fkm.imputation(data)
    
    logger.info("=> Time of Imputation \"" + fkm.getTime + "\"")
    
  }
  
}