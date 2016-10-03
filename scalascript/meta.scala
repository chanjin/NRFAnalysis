import java.io.{BufferedWriter, FileWriter}



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

val workspace ="/Users/chanjinpark/data/NRFdata/"
val metafile = Array("NRF2013Meta.csv", "NRF2014Meta.csv", "NRF2015Meta.csv").map(x => workspace + x)

def trim(s: String) = {
  val patterns = "[()[0-9]-/:]"
  s.replaceAll(patterns, "")
}

// 65개 컬럼, 21 means project id, 32 means area
val meta = sc.union(metafile.map(f => sc.textFile(f).map(s => {
  val arr = s.split(",")
  val areacode = (Array(31, 32, 33, 34), Array(35, 36, 38, 39, 41, 42, 44, 45, 47, 48), Array(50, 51, 53, 54, 56, 57, 59, 60, 62, 63))
  val mainarea = areacode._1.map(arr(_))
  val nat = areacode._2.map(arr(_))
  val sixT = areacode._3.map(arr(_))
  (arr(21), MetaData(arr(22),  mainarea.map(trim(_)), nat.map(trim(_)), sixT.map(trim(_))))
})))

val file = new java.io.File("data/temp/meta.txt")
val bw = new BufferedWriter(new FileWriter(file))
meta.foreach(m => bw.write(m._1 + "-" + m._2.toString +"\n"))
bw.close()