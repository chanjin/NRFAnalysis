package viz


/**
  * Created by chanjinpark on 2016. 9. 14..
  */
object ClassificationResult {

  def main(args: Array[String]): Unit = {

  }

  import org.apache.spark.mllib.evaluation.MulticlassMetrics

  import java.io._
  def saveMulticlassMetrics(dn: String, metrics: MulticlassMetrics, classes: Map[Int, String] ) = {
    val cm = metrics.confusionMatrix
    var f = new BufferedWriter(new FileWriter(new File(dn + "metrics-confusionmatrix.csv")))
    var m = metrics.confusionMatrix
    f.write((0 until m.numCols).map(i => classes.get(i)).mkString(",") + "\n")
    (0 until m.numRows).foreach(i =>
      f.write((0 until m.numCols).map(j => m(i, j)).mkString(",") + "\n")
    )
    f.close()

    f = new BufferedWriter(new FileWriter(new File(dn + "metrics-precision-recall.csv")))

    f.write("label, precision, recall\n")
    metrics.labels.foreach { l =>
      f.write(classes(l.toInt) + ", " + metrics.precision(l) + "," +  metrics.recall(l) + "\n")
    }
    f.close()

    metrics.labels.foreach { l =>
      print(classes(l.toInt) + ", " + metrics.precision(l) + "," +  metrics.recall(l) + "\n")
    }
  }


}
