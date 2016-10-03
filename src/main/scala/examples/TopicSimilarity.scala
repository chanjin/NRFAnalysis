package examples

/**
  * Created by chanjinpark on 2016. 9. 26..
  */
object TopicSimilarity {

  def getTopicTerms(f: String): Map[Int, List[(String, Double)]]= {
    val fname = f + "/topics_terms.csv"
    val tterms = scala.io.Source.fromFile(fname).getLines().
      drop(1).map(l => {
      val arr = l.split(",")
      (arr(0).toInt, (arr(1).trim(), arr(2).toDouble))
    }).toList.groupBy(_._1).map(g => (g._1, g._2.map(_._2).sortBy(-_._2)))
    tterms
  }

  def getTopicSimilarity(t1: List[(String, Double)], t2: List[(String, Double)]) = {

  }

  def main(args: Array[String]): Unit = {
    val tterms = getTopicTerms("ldaviz" + "2014")
    tterms.head._2.foreach(println)
  }

}
