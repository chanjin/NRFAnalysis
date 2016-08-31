package basic

import org.apache.spark.mllib.clustering.LDAModel

import scala.io._
import java.io._

object LDAVizHTML {
  def main(args: Array[String]): Unit = {
    val page =
      <html>
        <head>
          <title>Hello XHTML world</title>
        </head>
        <body>
          <h1>Hello world</h1>
          <p>
            <a href="scala-lang.org">Scala</a>
            talks XHTML</p>
        </body>
      </html>




    val file = new File("ldaviz/sample.html");
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(page.toString)
    bw.close()
  }
}

class LDAVizHTML( ldaModel: LDAModel,  vocabArray: Map[Int, String],  docs: Array[String],
                  area: Map[String, MetaData],  numTerms: Int,
                 docpath: String => String) {

  import org.apache.spark.mllib.clustering.DistributedLDAModel
  val distmodel = ldaModel.asInstanceOf[DistributedLDAModel]
  val topics = ldaModel.describeTopics(maxTermsPerTopic = numTerms)
  val numTopics = ldaModel.k

  def fileWrite(name: String, cont: String) = {
    val file = new File("ldadata/" + name);
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(cont)
    bw.close()
  }

  def generateCirclePacking() = {
    val json = "{\n\"name\": \"lda\", \"children\":[\n" +
      (0 until numTopics).map(i => {
        "{ \n\"name\" : \"topic " + i + "\", \n\"children\": [" +
          topics(i)._1.zip(topics(i)._2).map(tw =>
            "{ \"name\" : \"" + vocabArray(tw._1.toInt) + "\", \"size\": " + (tw._2 * 1000).toInt + " }").
            mkString(",") +
          "]\n}"
      }).mkString(",\n") + "\n]\n}"

    val writer = new PrintWriter(new File("ldaviz/lda.json"))
    writer.write(json)
    writer.close()
  }

  def isICT(did: Int) = {
    val a = area(docs(did))
    a.mainArea(0).equals("ICT·융합연구")
  }


  // CRB, Nat, 6T
  def summarizeDocs(doclist: Array[Long]) = {
    val summary = doclist.map(did => {
      val a = area(docs(did.toInt))
      (a.mainArea(0), a.mainArea(1), a.nationArea(0), a.sixTArea(0))
    }).foldLeft((Map[String, Int](), Map[String, Int](), Map[String, Int](), Map[String, Int]()))((res, as) => {
      (res._1 + (as._1 -> (res._1.getOrElse(as._1, 0) + 1)), res._2 + (as._2 -> (res._2.getOrElse(as._2, 0) + 1) ),
        res._3 + (as._3 -> (res._3.getOrElse(as._3, 0) + 1)), res._4 + (as._4 -> (res._4.getOrElse(as._4, 0) + 1)))
    })
    def makestr(m: Map[String, Int]) = m.toList.sortBy(-_._2).map(kv => kv._1 + " - " + kv._2).mkString(",\t")
    (makestr(summary._1), makestr(summary._2), makestr(summary._3), makestr(summary._4))
  }

  // topicsDetails(ldaModel, vocabArray, docs)
  def topicsDetails(topDocsPerTopic: Array[(Array[Long], Array[Double])]) = {
    (0 until numTopics).foreach(k => {

      val terms = topics(k)._1.zip(topics(k)._2).map { case (t, w) =>
        <tr>
          <td>
            {s"${vocabArray(t)}"}
          </td> <td>
          {f"$w%1.3f"}
        </td>
        </tr>
      }

      val doclist = topDocsPerTopic(k)._1.zip(topDocsPerTopic(k)._2).filter(dw => dw._2 > 0.1)
      val cntICT = doclist.map { case (d, w) => {
          if (isICT(d.toInt)) 1 else 0
        }}.sum

      val docsummary = summarizeDocs(doclist.map(_._1))

      val documents = doclist.map { case (d, w) =>
        <tr>
          <td>
            { if (isICT(d.toInt)) "ICT융합" else "-----" }
          </td>
          <td>
            <a href={"docs/" + docs(d.toInt) + ".html"}>
              {s"${docs(d.toInt)}"}
            </a>
          </td>
          <td>
            {f"$w%1.3f"}
          </td>
        </tr>
      }


      val detailpage =
        <html>
          <head>
            <title>
              {s"topic$k"}
            </title>
          </head>
          <body>
            <hr color="red">Topic #
              {s"$k"}
            </hr>

            <hr color="red">분류체계 연관성</hr>
            <table>
              <tr><td>{f"ICT 과제 수"}</td> <td> {s"$cntICT / ${doclist.length}, " + f"${cntICT.toDouble/doclist.length}%1.2f"} </td></tr>
              <tr><td>{s"학문분야 최종"}</td> <td>{ s"${docsummary._1}"} </td></tr>
              <tr><td>{s"학문분야 CRB"}</td> <td> {s"${docsummary._2}"} </td></tr>
              <tr><td>{s"국가과학기술표준분류"}</td><td> {s"${docsummary._3}"}</td></tr>
              <tr><td>{s"6T 기술분류"}</td><td>{s"${docsummary._4}"}</td></tr>
            </table>
            <hr color="red">Terms</hr>
            <table>
              {terms}
            </table>
            <hr color="red">Related Documents</hr>
            <table>
              {documents}
            </table>
          </body>
        </html>
      scala.xml.XML.save(s"ldaviz/$k.html", detailpage, "UTF-8")

      //val filecont =
    })
  }

  def getArea(did: Int) = {
    area(docs(did))
  }

  // documentsPage(ldaModel, vocabArray, docs)
  def documentsPage(doc2topics: Array[(Long, Array[Int], Array[Double])]) = {
    doc2topics.foreach { case (did, tids, ws) => {

      val s = docs(did.toInt)
      val docname = docpath(s)
      val areas = area(s)
      val arealist = {
        <tr>
          <td>연구과제명</td>
          <td>
            {areas.title}
          </td>
        </tr>
          <tr>
            <td>학문분야</td>
            <td>
              {areas.mainArea.mkString(", ")}
            </td>
          </tr>
          <tr>
            <td>국가과학기술표준분류</td>
            <td>
              {area(s).nationArea.mkString(",")}
            </td>
          </tr>
          <tr>
            <td>6T 기술분류</td>
            <td>
              {area(s).sixTArea.mkString(",")}
            </td>
          </tr>
      }

      val topiclist = tids.zip(ws).map { case (t, w) =>
        <tr>
          <td>
            {topics(t)._1.map(term => vocabArray(term)).mkString(",")}
          </td>
          <td>
            {f"$w%1.3f"}
          </td>
          <td>
            <a href={s"../$t.html"}>Topic #
              {s"$t"}
            </a>
          </td>
        </tr>
      }

      try {
        val docpage =
          <html>
            <head>
              <title>
                {s}
              </title>
            </head>
            <body>
              <hr color="red">Name:
                {s}
              </hr>
              <table>
                {arealist}
              </table>
              <hr color="red">Topics:
              </hr>
              <table>
                {topiclist}
              </table>
              <hr color="red">Original Text</hr>
              <pre>{scala.io.Source.fromFile(docname)("UTF-8")} </pre>
            </body>
          </html>

        scala.xml.XML.save(s"ldaviz/docs/$s.html", docpage, "UTF-8")
      } catch {
        case e: Exception => {
          println(docname); println(e); throw e
        }
      }
    }
    }
  }

  def topicsAllPage(): Unit = {
    val topicstr = (0 until numTopics).map(i => {
      val topic = topics(i)
      (i, s"#$i " + topic._1.zip(topic._2).map { case (t, w) => s"${vocabArray(t.toInt)}" }.mkString(","))
    })

    val topicpage =
      <html>
        <head>
          <title>LDA</title>
        </head>
        <body>
          <hr color="red">Topics</hr>
          <table>
            {topicstr.map { case (id, str) => <tr>
            <td>
              <a href={id + ".html"}>
                {str}
              </a>
            </td>
          </tr>
          }}
          </table>
        </body>
      </html>

    scala.xml.XML.save("ldaviz/ldaviz.html", topicpage, "UTF-8")


  }



  def generatePages() = {

    topicsAllPage()

    //val topic2docs : Map[Int, (Array[Long], Array[Double])] = distmodel.topDocumentsPerTopic(10).zipWithIndex.map(_.swap).toMap
    //val topicDetails = topics.zip(distmodel.topDocumentsPerTopic(10)).zipWithIndex.map(_.swap)
    val topDocsPerTopic = distmodel.topDocumentsPerTopic(50)
    topicsDetails(topDocsPerTopic)

    val docs2topics = distmodel.topTopicsPerDocument(20).collect()
    documentsPage(docs2topics)

    generateCirclePacking
  }

  //    topicsAllPage(topics1, vocabArray)
}