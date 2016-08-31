
```scala
package LDA

import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chanjinpark on 2016. 4. 7..
  */
object LDA {

  def main(args: Array[String]) : Unit = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    def parseProject(content: List[String]): Map[String, String] = {
      var kvlist = Map[String, String]()
      var (key, value) = ("", List[String]())
      content.foreach(l=> {
        if (l.contains(":")) {
          val pair = l.split(":").map(_.trim).toList
          if ( key.length > 0 ) {
            kvlist = kvlist + (key -> value.reverse.mkString(" "))
          }
          key = pair.head
          value = pair.tail
        }
        else {
          value = l.trim :: value
        }
      })
      if (content.nonEmpty) {
        kvlist = kvlist + (key -> value.reverse.mkString(" "))
      }

      var fields_ = List[String]()
      fields_ = fields_ ::: (if (kvlist.contains("NSF Program")) kvlist("NSF Program").split(",").toList else List[String]())
      fields_ = fields_ ::: (if (kvlist.contains("Fld Applictn")) kvlist("Fld Applictn").split(",").toList else List[String]())

      val fields = fields_.map(f => {
        if (f.length > 0 && Character.isDigit(f.charAt(0))) {
          f.substring(f.indexOf(" "), f.length).trim
        }
        else f.trim
      }).filter(_.length > 0).toList

      kvlist + ("fields" -> fields.mkString(","))
    }


    val inputdata = sc.wholeTextFiles("/Users/chanjinpark/DevProj/NSFAwards/data/*/*/*.txt").
      map(x => {
        val kv = parseProject(x._2.split("\n").toList)
        (kv("File"), kv("Title"),  kv("Abstract"), kv("fields"), x._1)
      })


    import org.apache.spark.rdd.RDD
    val docs = inputdata.map(x => x._1 + ": " + x._4 + ", " + x._5.substring(5).trim).collect()
    val corpus: RDD[String] = inputdata.map(x => x._2 + x._3)

    val tokenized = corpus.map(_.toLowerCase.split("\\s")).
        map(_.filter(_.length > 3).
          filter(_.forall(java.lang.Character.isLetter)))

    val termCounts: Array[(String, Long)] =
      tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

    val numStopwords = 20
    val vocabArray: Array[String] = termCounts.takeRight(termCounts.size - numStopwords).map(_._1).
      filter(s => !("where,only,there,then,into,have,should,will,following,same,must,other,than,over,some,they,like,such,since,takes," +
        "while,does,many,either,within,before,after,want,provide,about,simple,between,used,well,support,through," +
        "both,important,more,during,using,including,understand,proposed,understanding,conference,results,university,graduate,program,development," +
        "international,program,design,been,most").split(",").map(_.trim).contains(s))

    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    import org.apache.spark.mllib.linalg.{Vector, Vectors}
    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] =
      tokenized.zipWithIndex.map { case (tokens, id) =>
        val counts = new scala.collection.mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }

    //val docs2paths =
    // documents.first 문서 별로 각 term의 사용 횟수를 기록 (문서번호, (단어 벡터))로 구성됨
    // 단어 벡터는 sparse 벡터로서
    // (1.0, 0.0, 3.0)를 덴스벡터로는 [1.0, 0.0, 3.0], 스파스벡터로는 (3, [0, 2], [1.0, 3.0])로 표시됨. 크기, 인덱스, 값
    // res13: (Long, org.apache.spark.mllib.linalg.Vector) =
    // (0,(4167,[0,1,4,7,8,10,11,12,21,25,53,55,57,61,63,64,65,78,83,94,98,105,107,110,114,1

    import org.apache.spark.mllib.clustering.LDA

    // Set LDA parameters
    val numTopics = 10
    val lda = new LDA().setK(numTopics).setMaxIterations(20)

    val ldaModel = lda.run(documents)

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 20)
    topicIndices.take(3).foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach {
        case (term, weight) =>
          println(s"${vocabArray(term.toInt)}\t$weight")
      }
      println()
    }


    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")

    import org.apache.spark.mllib.clustering.DistributedLDAModel
    val distmodel = ldaModel.asInstanceOf[DistributedLDAModel]
    val docspertopic = distmodel.topDocumentsPerTopic(10)


    //LDAVizHTML.generatePages(ldaModel,vocabArray,docs)

    docspertopic.take(3).foreach {
      case (docids, docweights) =>
        println("TOPIC:")
        docids.zip(docweights).foreach {
          case (d, w) =>
            println(s"${w}:\t${docs(d.toInt)}")
        }
    }


    val topicsperdocument = distmodel.topTopicsPerDocument(3)
    topicsperdocument.take(3).foreach {
      case (doc, topics, w) => {
        println("DOC:" + s"${docs(doc.toInt)}")
        topics.zip(w).foreach {
          case (t, w) =>
            println(s"TOPIC ${t}\t${w}")
        }
      }
    }
  }
}

```