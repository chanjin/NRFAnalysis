package basic

import org.apache.spark.mllib.clustering.LDAModel

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


    import java.io._

    val file = new File("ldaviz/sample.html");
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(page.toString)
    bw.close()
  }


  def generateCirclePacking(topics: Array[(Array[Int], Array[Double])], vocabArray: Map[Int, String], numTopics: Int) = {
    val json = "{\n\"name\": \"lda\", \"children\":[\n" +
      (0 until numTopics).map(i => {
        "{ \n\"name\" : \"topic " + i + "\", \n\"children\": [" +
          topics(i)._1.zip(topics(i)._2).map(tw =>
            "{ \"name\" : \"" + vocabArray(tw._1.toInt) + "\", \"size\": " + (tw._2 * 1000).toInt + " }").
            mkString(",") +
          "]\n}"
      }).mkString(",\n") + "\n]\n}"

    import java.io.{File, PrintWriter}
    val writer = new PrintWriter(new File("ldaviz/lda.json"))
    writer.write(json)
    writer.close()
  }

  //val docRootDir = "/Users/chanjinpark/Dev/spark-1.6.0/docs/"
  val docRootDir = "/Users/chanjinpark/data/NRF2015/content/"

  def doc2pathname(s: String): String = docRootDir + s + ".txt"

  // topicsDetails(ldaModel, vocabArray, docs)
  def topicsDetails(topics: Array[(Array[Int], Array[Double])], numTopics: Int, topDocsPerTopic: Array[(Array[Long], Array[Double])], vocabArray: Map[Int, String], docs: Array[String]) = {
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

      val documents = topDocsPerTopic(k)._1.zip(topDocsPerTopic(k)._2).map { case (d, w) =>
        <tr>
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
    })
  }


  // documentsPage(ldaModel, vocabArray, docs)
  def documentsPage(topics: Array[(Array[Int], Array[Double])], doc2topics: Array[(Long, Array[Int], Array[Double])],
                    vocabArray: Map[Int, String], docs: Array[String], area: Map[String, (String, Array[String], Array[String], Array[String])]) = {
    doc2topics.foreach { case (did, tids, ws) => {

      val s = docs(did.toInt)
      val docname = doc2pathname(s)
      val areas = area(s)
      val arealist = {
        <tr>
          <td>연구과제명</td>
          <td>
            {areas._1}
          </td>
        </tr>
          <tr>
            <td>학문분야</td>
            <td>
              {areas._2.mkString(", ")}
            </td>
          </tr>
          <tr>
            <td>국가과학기술표준분류</td>
            <td>
              {area(s)._3.mkString(",")}
            </td>
          </tr>
          <tr>
            <td>6T 기술분류</td>
            <td>
              {area(s)._4.mkString(",")}
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

  def topicsAllPage(topics: Array[(Array[Int], Array[Double])], numTopics: Int, vocabArray: Map[Int, String], docs: Array[String]): Unit = {
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


  def generatePages(ldaModel: LDAModel, vocabArray: Map[Int, String], docs: Array[String], area: Map[String, (String, Array[String], Array[String], Array[String])]) = {
    import org.apache.spark.mllib.clustering.DistributedLDAModel
    val distmodel = ldaModel.asInstanceOf[DistributedLDAModel]
    val topics = ldaModel.describeTopics(maxTermsPerTopic = 10)
    val numTopics = ldaModel.k

    topicsAllPage(topics, numTopics, vocabArray, docs)

    //val topic2docs : Map[Int, (Array[Long], Array[Double])] = distmodel.topDocumentsPerTopic(10).zipWithIndex.map(_.swap).toMap
    //val topicDetails = topics.zip(distmodel.topDocumentsPerTopic(10)).zipWithIndex.map(_.swap)
    val topDocsPerTopic = distmodel.topDocumentsPerTopic(50)
    topicsDetails(topics, numTopics, topDocsPerTopic, vocabArray, docs)

    val docs2topics = distmodel.topTopicsPerDocument(20).collect()
    documentsPage(topics, docs2topics, vocabArray, docs, area)

    generateCirclePacking(topics, vocabArray, numTopics)
  }

  //    topicsAllPage(topics1, vocabArray)
}