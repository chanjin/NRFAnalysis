package basic

/**
  * Created by chanjinpark on 2016. 9. 6..
  */
case class MetaData(title: String, mainArea: Array[String], nationArea: Array[String], sixTArea: Array[String]) {
  override def toString: String = title + ":::" + mainArea.mkString(",") + ":::" +
    nationArea.mkString(",") +  ":::" +  sixTArea.mkString(",")
}
object MetaData {
  def apply(s: String): MetaData = {
    val attr = s.split(":::")
    new MetaData(attr(0), attr(1).split(","), attr(2).split(","), attr(3).split(","))
  }
}
