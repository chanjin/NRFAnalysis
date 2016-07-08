package LDA

import java.util.Properties

import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.util.CoreMap

/**
  * Created by chanjinpark on 2016. 4. 11..
  */
object LDALemmatized {

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


    def loadStopWords(path: String) = scala.io.Source.fromFile(path).getLines().toSet

    val stopWords = sc.broadcast(loadStopWords("./stopwords.txt")).value

    val docs = inputdata.map(x => x._1 + ": " + x._4 + ", " + x._5.substring(5).trim).collect()
    val corpus: RDD[String] = inputdata.map(x => x._2 + x._3)

    val lemmatized : RDD[Seq[String]] = corpus.mapPartitions(iter => {
      // createNLPPipeline
      val props = new Properties()
      props.put("annotators",  "tokenize, ssplit, pos, lemma") //"tokenize, ssplit, pos, lemma, ner, parse, dcoref"
      val pipeline = new StanfordCoreNLP(props)

      // lemmatize
      iter.map { case contents =>
        NSFAwards.lemmatize(contents, stopWords, pipeline).
          filter(s => !("research,researcher,participant,community,university,provide,use,student,problem,new,workshop,develop,hold,international," +
            "science,engineering,conference,area,engineering,group,study,analysis,program,project,work,system,undergraduate,technology,education," +
            "support,propose,design,result,effect,year,proposal,include,development,understand,understanding,important").
            split(",").map(_.trim).contains(s))
      }
    })

    val vocabArray = lemmatized.flatMap(x => x).distinct().collect()
    val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

    import org.apache.spark.mllib.linalg.{Vector, Vectors}
    // Convert documents into term count vectors
    val documents: RDD[(Long, Vector)] =
      lemmatized.zipWithIndex.map { case (tokens, id) =>
        val counts = new scala.collection.mutable.HashMap[Int, Double]()
        tokens.foreach { term =>
          if (vocab.contains(term)) {
            val idx = vocab(term)
            counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
          }
        }
        (id, Vectors.sparse(vocab.size, counts.toSeq))
      }


    import org.apache.spark.mllib.clustering.LDA

    // Set LDA parameters
    val numTopics = 20
    val lda = new LDA().setK(numTopics).setMaxIterations(25)
    val ldaModel = lda.run(documents)

    //LDAVizHTML.generatePages(ldaModel,vocabArray,docs)

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 10)
    topicIndices.take(3).foreach { case (terms, termWeights) =>
      println("TOPIC:")
      terms.zip(termWeights).foreach {
        case (term, weight) =>
          println(s"${vocabArray(term.toInt)}\t$weight")
      }
      println()
    }

    println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize + " words):")

    /*
    import org.apache.spark.mllib.clustering.DistributedLDAModel
    val distmodel = ldaModel.asInstanceOf[DistributedLDAModel]
    val docspertopic = distmodel.topDocumentsPerTopic(10)

    docspertopic.take(3).foreach {
      case (documents, docsWeights) =>
        println("TOPIC:")
        documents.zip(docsWeights).foreach {
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


    // ---------------------------------------------------------------------------

    import scala.io.Source
    val file1 = "/Users/chanjinpark/DevProj/NSFAwards/data/awards_2003/awd_2003_00/a0300025.txt"
    NSFAwards.parseProject(Source.fromFile(file1).getLines().toList).foreach(println)
    val txt = NSFAwards.parseProject(Source.fromFile(file1).getLines().toList)
    txt("Abstract")
    txt("Total Amt.")
    txt("Title")
    txt("File")

    //val files = getListOfFiles(path).map(f => (f, sc.textFile(f)))
    def getListOfFiles(dir: String):List[String] = {
      val d = new File(dir)
      if (d.exists && d.isDirectory) {
        d.listFiles.flatMap(f => {
          if (f.isFile) List(f.getAbsolutePath)
          else if (f.isDirectory) getListOfFiles(f.getAbsolutePath)
          else List[String]()
        }).toList
      }
      else {
        List[String]()
      }
    }
    */
  }

}
