package basic

import org.apache.log4j.{Level, Logger}
import org.apache.lucene.analysis.ko.morph.{MorphAnalyzer, PatternConstants}
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.linalg.{SparseVector, Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext

/**
  * Created by chanjinpark on 2016. 7. 7..
  */
// 4, 10, 10


trait PreProcessing extends Serializable {

  var stopWords: Set[String] = null
  val stopAreas: Set[String] = Set("0", "6T", "기타(위의 미래유망신기술(6T) 103개 세분류에 속하지 않는 기타 연구)")
  def replace(s: String) = s.replaceAll("[-()[0-9]]", "").split("/")

  /*
    21: 과제 ID    22: 과제 제목
    31: 학문분야(최종)	=> (5)ICT·융합연구    32: CRB(15년 분류기준)    33: RB(15년 분류기준)
    34: RB세분류(15년 분류기준)    35~48: 국가과학기술표준분류    50~63: 6T 기술분류
   */
  def getMetaData(sc: SparkContext, fs: Array[String]):  RDD[(String, MetaData)] = {
    def trim(s: String) = {
      val patterns = "[()[0-9]-/:]"
      s.replaceAll(patterns, "")
    }
    // 65개 컬럼, 21 means project id, 32 means area
    val meta = sc.union(fs.map(f => sc.textFile(f).map(s => {
      val arr = s.split(",")
      val areacode = (Array(31, 32, 33, 34), Array(35, 36, 38, 39, 41, 42, 44, 45, 47, 48), Array(50, 51, 53, 54, 56, 57, 59, 60, 62, 63))
      /*val mainarea = areacode._1.map(arr(_)).filter(s => !stopAreas.contains(s)).flatMap(replace(_)).filter(_.length > 0).distinct
      val nat = areacode._2.map(arr(_)).filter(s => !stopAreas.contains(s)).flatMap(replace(_)).filter(_.length > 0).distinct
      val sixT = areacode._3.map(arr(_)).filter(s => !stopAreas.contains(s)).flatMap(replace(_)).filter(_.length > 0).distinct
      */
      val mainarea = areacode._1.map(arr(_))
      val nat = areacode._2.map(arr(_))
      val sixT = areacode._3.map(arr(_))
      (arr(21), MetaData(arr(22),  mainarea.map(trim(_)), nat.map(trim(_)), sixT.map(trim(_))))
    })))

    meta
  }


  private def getInputData(sc: SparkContext, dir: Array[String]) = {
    val inputdata = sc.union(dir.map( d => {
      sc.wholeTextFiles(d + "*.txt").
        map(x => {
          val id = x._1.substring(x._1.lastIndexOf("/") + 1, x._1.lastIndexOf("."))
          val lines = x._2.split("\n")
          var words = Array[String]()
          try {
            words = lines.flatMap(l => vocabWords(preprocess(l)))
          } catch {
            case e: Exception => {
              println(e); println(id); println(lines)
            }
          }
          (id, words)
        })
      }
    ))

    val stopWordsSet = Set("of", "and", "in", "the", "Mixed", "for", "근간한", "대한", "활용", "관련", "가능", "연구", "개발", "통한",
      "제시", "제공", "이용", "적용", "다양", "0", "의한", "ET", "중요", "to", "be", "by", "with", "will", "on", "부터", "및",
      "위한", "이용한", "개발", "관한", "미치는", "기반의", "활용한", "규명", "새로운", "기반", "유래", "기술", "시스템", "따른", "II",
    "방안", "기법")
    stopWords = sc.broadcast(stopWordsSet).value

    inputdata
  }

  def getCorpus(sc: SparkContext, dir: Array[String], meta: Map[String, MetaData]) = {
    val inputdata = getInputData(sc, dir)
    val docs = inputdata.keys//.map(s => s.substring(0, s.lastIndexOf(".txt")))
    val corpus = inputdata.map{
      case (id, strs) => {
        (strs ++ preprocess(meta(id).title).split(" ")).filter(s => !stopWords.contains(s) && s.length != 0)
      }
    }
    (docs, corpus)
  }

  def getMatrix(corpus: RDD[Array[String]]) = {
    val vocab = corpus.flatMap(x => x).distinct.collect.zipWithIndex.toMap
    val matrix: RDD[(Long, Vector)] = corpus.zipWithIndex.map {
      case (tokens, docid) => {
        //val counts = new scala.collection.mutable.HashMap[Int, Double]()
        val vec = tokens.foldLeft(Map[Int, Double]())((res, t) => {
          val vocabid = vocab(t)
          res + (vocabid -> (res.getOrElse(vocabid, 0.0) + 1.0))
        })
        //val (indices, values) = vec.keys
        (docid, new SparseVector(vocab.size, vec.keys.toArray, vec.values.toArray))
      }
    }
    (vocab, matrix)
  }


  import org.apache.lucene.analysis.ko.morph._
  import scala.collection.JavaConversions._

  def vocabWords(source: String): Array[String] = {
    val ma = new MorphAnalyzer
    val words = source.split(" ").filter(_.length > 1).flatMap(s => {
      val morphemes = ma.analyze(s.trim).map(s => s.asInstanceOf[AnalysisOutput])
      //val o = morphemes.filter(_.getPos == PatternConstants.POS_NOUN).map(r => r.getStem)
      val o = morphemes.filter(m => m.getPos == PatternConstants.POS_NOUN) //'N')

      if (o.length > 1) {
        List(o.head.getStem).filter(_.length > 1)
      }
      else {
        o.flatMap(out => {
          val cn = out.getCNounList.map(_.getWord)
          val stem = out.getStem
          if (cn.length > 1) cn
          else List(stem)
        }).filter(_.length > 1)
      }
    })

    words
  }


  /*val pattern = "[－()\\-,()/?.\uDBC1\uDE5B￭\uF06C\uF09E▷\uDB80\uDEEF<\uDEEF\uF0A0１‘２\uF0D7･～\uF06D▶\uF09F\uDB80\uDEFB:" +
    "\uDB80\uDEEB\uD835\uDF70~=---\";;「ㆍ’'“”／·•⦁▸◎>○\uF061：　╸∎▪◦˚◼︎●■→*（(((，茶．·＜＞+①②③➌④" +
    "３□）)))九六補瀉法︎\uF02D\uDB80\uDEEE龍脈\uDB80\uDEB1\uDB80\uDEB2\uDB80\uDEB3捻轉ㅃ∙㉮ㅇ///◆\uD835\uDEC3＊\uDBFA\uDF16]" //
    */
  val pattern = "[^(가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9)]"

  def preprocess(s: String) = {
    s.replaceAll(pattern, " ").replaceAll("[()/:]", " ")//.replaceAll(pattern2, " ").replaceAll("]", " ")
  }


}

trait TFIDF {
  def getMatrix(corpus: RDD[Array[String]]) : (RDD[Vector], HashingTF) = {
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(corpus.map(c => c.toIndexedSeq))

    import org.apache.spark.mllib.feature.IDF
    tf.cache()
    val idf = new IDF().fit(tf)
    (idf.transform(tf), hashingTF)
  }
}

object PreprocessingTest extends PreProcessing {
  def main(args: Array[String]) : Unit = {
    println(preprocess("aaaaa한글ab-b123*b)ccc•⦁▸◎dddii) 통합 유무선네트워크(Integrated Ship Area Network: i-SAN) 및 W"))
  }
}
