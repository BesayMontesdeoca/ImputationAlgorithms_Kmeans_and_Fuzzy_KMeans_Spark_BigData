package imputationClustering



import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.collection.mutable.ArrayBuffer
import org.apache.log4j.Logger
import utils.keel.KeelParser

/**
 * Distributed kNN class.
 *
 *
 * @author Jesus Maillo
 */

class KNNImputation(train: RDD[Vector], test: RDD[Vector], k: Int, distanceType: Int, numClass: Int, numFeatures: Int, numPartitionMap: Int, numReduces: Int, numIterations: Int, maxWeight: Double, isCategorical: Array[Boolean]) extends Serializable {

  //Count the samples of each data set and the number of classes
  private val numSamplesTrain = train.count()
  private val numSamplesTest = test.count()

  //Setting Iterative MapReduce
  private var inc = 0
  private var subdel = 0
  private var topdel = 0
  private var numIter = numIterations
  private def broadcastTest(test: Array[Vector], context: SparkContext) = context.broadcast(test)

  //Getters
  def getTrain: RDD[Vector] = train
  def getTest: RDD[Vector] = test
  def getK: Int = k
  def getDistanceType: Int = distanceType
  def getNumClass: Int = numClass
  def getNumFeatures: Int = numFeatures
  def getNumPartitionMap: Int = numPartitionMap
  def getNumReduces: Int = numReduces
  def getMaxWeight: Double = maxWeight
  def getNumIterations: Int = numIterations

  /**
   * Initial setting necessary. Auto-set the number of iterations and load the data sets and parameters.
   *
   * @return Instance of this class. *this*
   */
  def setup(): KNNImputation = {

    //Starting logger
    var logger = Logger.getLogger(this.getClass())

    //Setting Iterative MapReduce
    var weightTrain = 0.0
    var weightTest = 0.0

    numIter = 0
    if (numIterations == -1) { //Auto-setting the minimum partition test.
      weightTrain = (8 * numSamplesTrain * numFeatures) / (numPartitionMap * 1024.0 * 1024.0)
      weightTest = (8 * numSamplesTest * numFeatures) / (1024.0 * 1024.0)
      if (weightTrain + weightTest < maxWeight * 1024.0) { //It can be run with one iteration
        numIter = 1
      } else {
        if (weightTrain >= maxWeight * 1024.0) {
          logger.error("=> Train wight bigger than lim-task. Abort")
          System.exit(1)
        }
        numIter = (1 + (weightTest / ((maxWeight * 1024.0) - weightTrain)).toInt)
      }

    } else {
      numIter = numIterations
    }

    logger.info("=> NumberIterations \"" + numIter + "\"")

    inc = (numSamplesTest / numIter).toInt
    subdel = 0
    topdel = inc
    if (numIterations == 1) { //If only one partition
      topdel = numSamplesTest.toInt + 1
    }

    this

  }

  /**
   * Imputation kNN
   *
   * @return RDD[Vector]. 
   */
  def imputation(): RDD[Vector] = {
    val testWithKey = test.zipWithIndex().map { line => (line._2.toInt, line._1) }.sortByKey().cache
    var logger = Logger.getLogger(this.getClass())
    var testBroadcast: Broadcast[Array[Vector]] = null
    var output: RDD[Vector] = null

    for (i <- 0 until numIter) {

      //Taking the iterative initial time.
      val timeBegIterative = System.nanoTime

      if (numIter == 1)
        testBroadcast = broadcastTest(test.collect, test.sparkContext)
      else
        testBroadcast = broadcastTest(testWithKey.filterByRange(subdel, topdel).map(line => line._2).collect, testWithKey.sparkContext)

      if (output == null) {
        output = testWithKey.join(train.mapPartitions(split => knn(split, testBroadcast, subdel)).reduceByKey(combine)).map(sample => imputeWithNeighbors(sample)).cache
      } else {
        output = output.union(testWithKey.join(train.mapPartitions(split => knn(split, testBroadcast, subdel)).reduceByKey(combine)).map(sample => imputeWithNeighbors(sample))).cache
      }
      output.count

      //Update the pairs of delimiters
      subdel = topdel + 1
      topdel = topdel + inc + 1
      testBroadcast.destroy
    }

    output

  }

  /**
   * Calculate the K nearest neighbor from the test set over the train set.
   *
   * @param iter Iterator of each split of the training set.
   * @param testSet The test set in a broadcasting
   * @param subdel Int needed for take order when iterative version is running
   * @return K Nearest Neighbors for this split
   */
  def knn[T](iter: Iterator[Vector], testSet: Broadcast[Array[Vector]], subdel: Int): Iterator[(Int, Array[(Double, Vector)])] = {
    // Initialization
    var train = new ArrayBuffer[Vector]
    val size = testSet.value.length

    var dist: Distance.Value = null
    //Distance MANHATTAN or EUCLIDEAN
    if (distanceType == 1)
      dist = Distance.Manhattan
    else
      dist = Distance.Euclidean

    //Join the train set
    while (iter.hasNext)
      train.append(iter.next)

    var knnMemb = new KNN_AL(train, k, dist, numClass)

    var auxSubDel = subdel
    var result = new Array[(Int, Array[(Double, Vector)])](size)

    for (i <- 0 until size) {
      result(i) = (auxSubDel, knnMemb.neighbors(testSet.value(i)))
      auxSubDel = auxSubDel + 1
    }

    result.iterator

  }

  /**
   * Join the result of the map taking the nearest neighbors.
   *
   * @param mapOut1 A element of the RDD to join
   * @param mapOut2 Another element of the RDD to join
   * @return Combine of both element with the nearest neighbors
   */
  def combine(mapOut1: Array[(Double, Vector)], mapOut2: Array[(Double, Vector)]): Array[(Double, Vector)] = {

    var itOut1 = 0
    var itOut2 = 0
    var out: Array[(Double, Vector)] = new Array[(Double, Vector)](k)

    var i = 0
    while (i < k) {
      if (mapOut1(itOut1)._1 <= mapOut2(itOut2)._1) { // Update the matrix taking the k nearest neighbors
        out(i) = mapOut1(itOut1)
        if (mapOut1(itOut1)._1 == mapOut2(itOut2)._1) {
          i += 1
          if (i < k) {
            out(i) = mapOut2(itOut2)
            itOut2 = itOut2 + 1
          }
        }
        itOut1 = itOut1 + 1

      } else {
        out(i) = mapOut2(itOut2)
        itOut2 = itOut2 + 1
      }
      i += 1
    }

    out
  }

  def imputeWithNeighbors(sample: (Int, (Vector, Array[(Double, Vector)]))): Vector = {
    val pointMV = sample._2._1
    val neighbours = sample._2._2.map(_._2)
    val indices = getIndices(pointMV)
    var ind = -1
    val res = pointMV.toArray.map { atr =>
      ind += 1
      if (atr == 0.0 && !indices.contains(ind)) {
        if (isCategorical(ind)) {
          val v = Array.fill(k)(0D)
          for (j <- 0 until k) {
            v(ind) = neighbours(j)(ind)
          }
          val votes = v.map((_, 1)).groupBy(identity).mapValues(_.length).max
          votes._1._1
        } else {
          var sum = 0.0
          for (j <- 0 until k) {
            sum += neighbours(j)(ind)
          }
          sum / k
        }

      } else {
        atr
      }
    }
    Vectors.dense(res)
  }
  
  private def getIndices(point: Vector): Array[Int] = {
    point.toString.split(Array('[', ']'))(1).split(",").map(_.toInt)
  }

}

/**
 * Distributed kNN class.
 *
 *
 * @author Jesus Maillo
 */
object KNNImputation {
  /**
   * Initial setting necessary.
   *
   * @param train Data that iterate the RDD of the train set
   * @param test The test set in a broadcasting
   * @param k number of neighbors
   * @param distanceType MANHATTAN = 1 ; EUCLIDEAN = 2
   * @param converter Dataset's information read from the header
   * @param numPartitionMap Number of partition. Number of map tasks
   * @param numReduces Number of reduce tasks
   * @param numIterations Autosettins = -1. Number of split in the test set and number of iterations
   */
  def setup(train: RDD[Vector], test: RDD[Vector], k: Int, distanceType: Int, numClass: Int, numFeatures: Int, numPartitionMap: Int, numReduces: Int, numIterations: Int, maxWeight: Double, isCategorical: Array[Boolean]) = {
    new KNNImputation(train, test, k, distanceType, numClass, numFeatures, numPartitionMap, numReduces, numIterations, maxWeight, isCategorical).setup()
  }
}