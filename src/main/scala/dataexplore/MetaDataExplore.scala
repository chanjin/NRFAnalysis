package dataexplore

import basic.MetaData

/**
  * Created by chanjinpark on 2016. 9. 2..
  */
object MetaDataExplore {

  def main(args: Array[String]): Unit = {
    val dir = "/Users/chanjinpark/GitHub/NRFAnalysis/"
    val meta = {
      scala.io.Source.fromFile(dir + "data/meta.txt").getLines().map(l => {
        val id = l.substring(0, l.indexOf("-"))
        val meta = MetaData(l.substring(l.indexOf("-") + 1))
        (id, meta)
      }).toMap
    }

    val crb = meta.groupBy(_._1.substring(0, 4).toInt).
      map(m => (m._1, m._2.map(_._2.mainArea(1)).toList.distinct.sortWith((x, y) => x.compareTo(y) < 0)))

    crb.foreach(println)
    crb(2013).diff(crb(2014)) // 2014년에 삭제  없음      List()
    crb(2014).diff(crb(2013)) // 2014년에 3개 추가됨      List(기초의학, 응용의학, 정보기술융합)
    crb(2014).diff(crb(2015)) // 2015년에 1개 삭제됨      List(의학)
    crb(2015).diff(crb(2014)) // 2015년에 1개 추가됨      List(정보전자소재융합)

  }
}
