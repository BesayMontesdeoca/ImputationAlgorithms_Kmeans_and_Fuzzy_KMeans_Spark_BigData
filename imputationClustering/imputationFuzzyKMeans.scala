package imputationClustering

import scala.collection.mutable.ArrayBuffer

import org.apache.spark.Logging
import org.apache.spark.annotation.{ Experimental, Since }
import org.apache.spark.mllib.linalg.{ Vector, Vectors }
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.random.XORShiftRandom

class imputationFuzzyKmeans(
    val k: Int,
    val m: Double,
    val maxIterations: Int,
    val seed: Int,
    val epsilon: Double,
    val parserMV: parserMV,
    val path_to_output: String) extends Serializable {

  var time: Double = 0.0

  /**
   * Imputation through the information of the centroids and the degrees of membership.
   */
  def imputation(data: RDD[Vector]): Unit = {

    val initStartTime = System.nanoTime()

    val centers = runAlgorithm(data)

    val impData = data.map { instance =>
      if (instance.numActives != parserMV.numFeatures) {
        impItem(centers, instance)
      } else {
        instance.toArray
      }
    }
    time = (System.nanoTime() - initStartTime) / 1e9
    impData.map(_.mkString(",")).coalesce(1, shuffle = true).saveAsTextFile(path_to_output)
  }
  
  /**
   * Implementation of Fuzzy-K-Means algorithm.
   */
  private def runAlgorithm(data: RDD[Vector]): Array[Vector] = {
    val sc = data.sparkContext

    val centers = initRandom(data)
    var converged = false
    var cost = 0.0
    var iteration = 0

    while (iteration < maxIterations && !converged) {
      val costAccum = sc.accumulator(0.0)
      val bcCenters = sc.broadcast(centers)

      val bcM = sc.broadcast(m)

      val newCenters = data.mapPartitions { points =>
        val thisCenters = bcCenters.value
        val dims = thisCenters.head.size
        val m = bcM.value

        val sums = Array.fill(k)(Vectors.zeros(dims))
        val fuzzyCounts = Array.fill(k)(0.0)

        points.foreach { point =>
          val (mbrpDegree, distances) = degreesOfMembership(thisCenters, point, m)
          mbrpDegree.zipWithIndex.
            filter(_._1 > epsilon * epsilon).
            foreach { degreeWithIndex =>
              val (deg, ind) = degreeWithIndex
              costAccum.add(deg * distances(ind))
              val sum = sums(ind)
              BLAS.axpy(deg, point, sum)
              fuzzyCounts(ind) += deg
            }
        }

        fuzzyCounts.indices.filter(fuzzyCounts(_) > 0).map(j => (j, (sums(j), fuzzyCounts(j)))).iterator
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
  private def initRandom(data: RDD[Vector]): Array[Vector] = {
    val rddNoMV = data.filter { x => x != -1 }
    rddNoMV.takeSample(false, k, seed)
  }
  
  /**
   * Returns the degree of membership of the point to each of the clusters
   * along with the array of distances from the point to each centroid
   */
  private def degreesOfMembership(centers: Array[Vector], point: Vector, fuzzifier: Double): (Array[Double], Array[Double]) = {
    val distances = centers.map { center => math.pow(distanceMV(center, point), -2 / (fuzzifier - 1)) }
    val sumDistances = distances.sum
    (distances.map { dis => dis / sumDistances }, distances)
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
   * Fill the vector values according to the information of the centroids and the degrees of membership.
   */
  private def impItem(centers: Array[Vector], point: Vector): Array[Double] = {
    val degrees = degreesOfMembership(centers, point, m)._1
    var i = -1
    val indices = getIndices(point)
    point.toArray.map { atr =>
      i += 1
      if (atr == 0.0 && !indices.contains(i)) {
        val dataImp = centers.zipWithIndex.map { center_ind =>
          val (center, ind) = center_ind
          center(i) * degrees(ind)
        }.sum
        
        if(parserMV.isAtrCategorical(i)){
          parserMV.getCategoricalLabel(i, dataImp)
        }else{
          dataImp
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