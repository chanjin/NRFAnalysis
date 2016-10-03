/**
  * Created by chanjinpark on 2016. 9. 25..
  */

case class MetaData(title: String, mainArea: Array[String], nationArea: Array[String], sixTArea: Array[String]) {
  override def toString: String = title + ":::" + mainArea.mkString(",") + ":::" +
    nationArea.mkString(",") +  ":::" +  sixTArea.mkString(",")
}
object MetaData {
  def apply(s: String): MetaData = {
    val attr = s.split(":::")
    if ( attr.length < 1 ) println(attr)
    new MetaData(attr(0), attr(1).split(","), attr(2).split(","), attr(3).split(","))
  }
}

val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"
val meta = {
  scala.io.Source.fromFile(dir + "data/meta.txt").getLines().map(l => {
    val id = l.substring(0, l.indexOf("-"))
    val meta = MetaData(l.substring(l.indexOf("-") + 1))
    (id, meta)
  }).toMap
}

/* CRB Values */
val crb = meta.groupBy(_._1.substring(0, 4).toInt).
  map(m => (m._1, m._2.map(_._2.mainArea(1)).toList.distinct.sortWith((x, y) => x.compareTo(y) < 0)))
crb.foreach(println)
crb.map(x => (x._1, x._2.size)).foreach(println)

println("2013 only removed")
crb(2013).diff(crb(2014)).foreach(println)
println("2014 added")
crb(2014).diff(crb(2013))

println("2014 only removed")
crb(2014).diff(crb(2015)).foreach(println)
println("2015 added")
crb(2015).diff(crb(2014)).foreach(println)

val projbycrb = meta.groupBy(_._1.substring(0, 4).toInt).map(m => (m._1, m._2.groupBy(_._2.mainArea(1)).map(x => (x._1, x._2.size))))


