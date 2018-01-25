package imputationClustering

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Logging
import org.apache.spark.annotation.{ Experimental, Since }
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom
import scala.sys.process._

class imputationKmeans(
    val k: Int,
    val maxIterations: Int,
    val seed: Int,
    val epsilon: Double,
    val parserMV: parserMV,
    val path_to_output: String) extends Serializable {

  private var time: Double = 0.0

  /**
   * Basic imputation through the information of the centroids
   */
  def imputation(data: RDD[Vector]): Unit = {

    val initStartTime = System.nanoTime()

    // Clustering stage
    val centers = runAlgorithm(data)

    // Imputation
    val impData = data.map { instance =>
      if (instance.numActives != parserMV.numFeatures) {
        impItem(centers, instance)
      } else {
        instance.toArray
      }
    }
    time = (System.nanoTime() - initStartTime) / 1e9

    // Save data
    impData.map(_.mkString(",")).coalesce(1, shuffle = true).saveAsTextFile(path_to_output)
  }

  /**
   * Imputation using the KNN algorithm
   */
  def imputationKNN(data: RDD[Vector], impK: Int, numPartitionMap: Int, numReduces: Int, numIterations: Int): Unit = {

    val initStartTime = System.nanoTime()

    // Clustering stage
    val centers = runAlgorithm(data)

    // Imputation
    var dataImp: RDD[Vector] = null
    val sc = data.sparkContext

    for (i <- 0 until k) {
      val dataCluster = data.filter { point => findClosest(centers, point)._1 == i }
      val data_NotMv = dataCluster.filter(_.numActives == parserMV.numFeatures)
      val data_MV = dataCluster.filter(_.numActives != parserMV.numFeatures)
      val knn = KNNImputation.setup(data_NotMv,
        data_MV,
        impK,
        0,
        parserMV.numClass,
        parserMV.numFeatures,
        numPartitionMap,
        numReduces,
        numIterations,
        0.0,
        parserMV.isCategorical)
      if (dataImp == null) {
        dataImp = knn.imputation()
      } else {
        dataImp = dataImp ++ knn.imputation()
      }
    }
    dataImp = data.filter(_.numActives == parserMV.numFeatures) ++ dataImp
    time = (System.nanoTime() - initStartTime) / 1e9

    // Save data
    dataImp.map(_.toArray).map(_.mkString(",")).coalesce(1, shuffle = true).saveAsTextFile(path_to_output)
  }

  /**
   * Implementation of K-Means algorithm.
   */
  private def runAlgorithm(data: RDD[Vector]): Array[Vector] = {

    val sc = data.sparkContext

    val centers = initRandom(data)

    var converged = false
    var cost = 0.0
    var iteration = 0

    // Execute iterations of Lloyd's algorithm until converged
    while (iteration < maxIterations && !converged) {
      val costAccum = sc.accumulator(0.0)
      val bcCenters = sc.broadcast(centers)

      // Find the new centers
      val newCenters = data.mapPartitions { points =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.size

        val sums = Array.fill(thisCenters.length)(Vectors.zeros(dims))
        val counts = Array.fill(thisCenters.length)(0L)

        points.foreach { point =>
          val (bestCenter, cost) = findClosest(thisCenters, point)
          costAccum.add(cost)
          val sum = sums(bestCenter)
          BLAS.axpy(1.0, point, sum)
          counts(bestCenter) += 1
        }

        counts.indices.filter(counts(_) > 0).map(j => (j, (sums(j), counts(j)))).iterator
      }.reduceByKey {
        case ((sum1, count1), (sum2, count2)) =>
          BLAS.axpy(1.0, sum2, sum1)
          (sum1, count1 + count2)
      }.mapValues {
        case (sum, count) =>
          BLAS.scal(1.0 / count, sum)
          sum
      }.collectAsMap()

      bcCenters.unpersist(blocking = false)

      // Update the cluster centers and costs
      converged = true
      newCenters.foreach {
        case (j, newCenter) =>
          if (converged && distanceMV(newCenter, centers(j)) > epsilon * epsilon) {
            converged = false
          }
          centers(j) = newCenter
      }

      cost = costAccum.value
      iteration += 1
    }
    centers
  }

  /**
   * Initialize a set of cluster centers at random.
   */
  def initRandom(data: RDD[Vector]): Array[Vector] = {
    val rddNoMV = data.filter { x => x.numActives == parserMV.numFeatures }
    rddNoMV.takeSample(false, k, seed)
  }

  /**
   * Returns the index of the closest center to the given point, as well as the squared distance.
   */
  private def findClosest(centers: TraversableOnce[Vector], point: Vector): (Int, Double) = {
    var bestDistance = Double.PositiveInfinity
    var bestIndex = 0
    var i = 0
    centers.foreach { center =>
      val distance: Double = distanceMV(center, point)
      if (distance < bestDistance) {
        bestDistance = distance
        bestIndex = i
      }
      i += 1
    }
    (bestIndex, bestDistance)
  }

  /**
   * Returns the Euclidean distance squared between two vectors 
   * taking into account the presence of the missing values.
   */
  private def distanceMV(center: Vector, point: Vector): Double = {
    if (point.numActives == parserMV.numFeatures) {
      math.sqrt(Vectors.sqdist(center, point))
    } else {
      val indices = getIndices(point)
      var ind = 0
      var sum: Double = 0.0
      point.toArray.foreach { atr =>
        if (atr == 0.0 && !indices.contains(ind)) {
          sum += 1.0
        } else {
          sum += math.pow(center(ind) - atr, 2)
        }
        ind += 1
      }
      math.sqrt(sum)
    }
  }

  /**
   * Fill the vector values according to the information of the centroids.
   */
  private def impItem(centers: Array[Vector], point: Vector): Array[Double] = {
    val indices = getIndices(point)
    val center = centers(findClosest(centers, point)._1)
    var ind = -1
    point.toArray.map { atr =>
      ind += 1
      if (atr == 0.0 && !indices.contains(ind)) {
        if (parserMV.isAtrCategorical(ind)) {
          parserMV.getCategoricalLabel(ind, center(ind))
        } else {
          center(ind)
        }
      } else {
        atr
      }
    }
  }

  /**
   * Returns the indexes of the Sparse Vector object.
   */
  private def getIndices(point: Vector): Array[Int] = {
    point.toString.split(Array('[', ']'))(1).split(",").map(_.toInt)
  }

  /**
   * Returns the execution time of the algorithm. 
   */
  def getTime = time

}