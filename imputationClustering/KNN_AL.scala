package imputationClustering

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.collection.mutable.ArrayBuffer

/** K Nearest Neighbors algorithms. 
 *  
 * @param train Traning set
 * @param k Number of neighbors
 * @param distanceType Distance.Manhattan or Distance.Euclidean
 * @param numClass Number of classes
 * @author sergiogvz
 */
class KNN_AL(val train: ArrayBuffer[Vector], val k: Int, val distanceType: Distance.Value, val numClass: Int) {

  val nFeatures = train(0).size
  
  /** Calculates the k nearest neighbors.
   *
   * @param x Test sample
   * @return Distance and Vector of nearest neighbors
   */
  def neighbors(x: Vector): Array[(Double, Vector)] = {
    var nearest = Array.fill(k)(-1)
    var distA = Array.fill(k)(0.0d)
    val size = train.length

    for (i <- 0 until size) { //for instance of the training set
      var dist: Double = Distance(train(i), x, distanceType, nFeatures)
      if (dist > 0d) {
        var stop = false
        var j = 0
        while (j < k && !stop) { //Check if it can be inserted as NN
          if (nearest(j) == (-1) || dist <= distA(j)) {
            for (l <- ((j + 1) until k).reverse) { //for (int l = k - 1; l >= j + 1; l--)
              nearest(l) = nearest(l - 1)
              distA(l) = distA(l - 1)
            }
            nearest(j) = i
            distA(j) = dist
            stop = true
          }
          j += 1
        }
      }
    }

    var out = new Array[(Double, Vector)](k)
    for (i <- 0 until k) {
      out(i) = (distA(i), train(nearest(i)))
    }
    out
  }

}

/** Factory to compute the distance between two instances.
 *  
 * @author sergiogvz
 */
object Distance extends Enumeration {
  val Euclidean, Manhattan = Value

  /** Computes the (Manhattan or Euclidean) distance between instance x and instance y.
   * The type of the distance used is determined by the value of distanceType.
   *
   * @param x instance x
   * @param y instance y
   * @param distanceType type of the distance used (Distance.Euclidean or Distance.Manhattan)
   * @return Distance
   */
  def apply(x: Vector, y: Vector, distanceType: Distance.Value, nFeatures: Int) = {
    distanceType match {
      case Euclidean => euclidean(x, y, nFeatures)
      case Manhattan => manhattan(x, y)
      case _         => euclidean(x, y, nFeatures)
    }
  }

  /** Computes the Euclidean distance between instance x and instance y.
   * The type of the distance used is determined by the value of distanceType.
   *
   * @param x instance x
   * @param y instance y
   * @return Euclidean distance
   */
  private def euclidean(pointNMV: Vector, point: Vector, nFeatures: Int): Double = {
    if(point.numActives == nFeatures){
      math.sqrt(Vectors.sqdist(pointNMV, point))
    }else{
      val indices = getIndices(point)
      var ind = 0
      var sum: Double = 0.0
      point.toArray.foreach { atr => 
        if(atr == 0.0 && !indices.contains(ind)){
          sum += 1.0
        }else{
           sum += math.pow(pointNMV(ind) - atr, 2)
        }
        ind += 1
      }
      math.sqrt(sum)
    }

  }

  /** Computes the Manhattan distance between instance x and instance y.
   * The type of the distance used is determined by the value of distanceType.
   *
   * @param x instance x
   * @param y instance y
   * @return Manhattan distance
   */
  private def manhattan(x: Vector, y: Vector) = {
    var sum = 0.0
    val size = x.size

    for (i <- 0 until size) sum += Math.abs(x(i) - y(i))

    sum.toFloat
  }
  
  private def getIndices(point: Vector): Array[Int] = {
    point.toString.split(Array('[', ']'))(1).split(",").map(_.toInt)
  }

}