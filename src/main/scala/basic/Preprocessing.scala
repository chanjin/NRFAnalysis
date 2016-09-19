package basic

import org.apache.log4j.{Level, Logger}
import org.apache.lucene.analysis.ko.morph.{MorphAnalyzer, PatternConstants}
import org.apache.spark.mllib.feature.{HashingTF, IDFModel}
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
  var stopDocs: Set[String] = null
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
    // Preprocess 함수 후에 써야할 것으로 보임
    val replaceMap = Map("ENERGY" -> "에너지", "MODEL" -> "모델", "CANCER" -> "암", "DISEASE" -> "질환", "ANALYSIS" -> "분석", "SYSTEM" -> "시스템",
      "THERAPY" -> "치료법", "STEM" -> "줄기", "PROTEIN" -> "단백질", "BRAIN" -> "뇌", "NETWORK" -> "네트워크", "LEARNING" -> "학습",
      "SMART" -> "스마트", "COMPUTER" -> "컴퓨터", "MAGNETIC"->"마그네틱", "SIGNAL" -> "시그널", "DATA" -> "데이터", "SENSOR" -> "센서",
      "DESIGN" -> "디자인", "MEMORY" -> "메모리", "MODELING" -> "모델링", "VIDEO" -> "비디오", "OPTIMIZATION" -> "최적화",
      "SOCIAL" -> "소셜", "VIRTUAL" -> "가상", "INFORMATION" -> "정보", "MULTIMEDIA" -> "멀티미디어", "ALGORITHM" -> "알고리즘",
      "WEB"->"웹", "LOCATION"->"위치", "RECOGNITION" -> "인지", "CONTROL" -> "제어", "SECURITY" -> "보안", "WEARABLE" -> "착용형",
      "MINING" -> "마이닝", "CONTEXT"->"컨텍스트", "INTELLIGENT" -> "지능", "SIMULATION"->"시물레이션", "WIRELESS"->"무선", "ARCHITECTURE" -> "아키텍처",
      "INTERFACE" -> "인터페이스", "DISTRIBUTED"->"분산", "DISTRIBUTION"->"분산", "CLOUD" -> "클라우드", "CELL" -> "세", "CELLS" -> "세포",
      "BIO"->"바이오", "DATABASE"->"데이터베이스", "PROCESSING"->"프로세싱", "CHANGE"->"변화", "CLIMATE"->"기후", "EVOLUTION"->"진화", "STUDY"->"연구",
      "STRUCTURE"->"구조", "CARBON"->"탄소", "AIR"->"에어", "GENE"->"유전자", "CULTURE"->"문화", "MACHINE"->"기계", "COMPUTING"->"컴퓨팅",
      "DETECTION"->"탐지", "INTERNET"->"인터넷", "PARALLEL"->"병렬", "MOBILE"->"모바일", "SOFTWARE"->"소프트웨어", "DEVICE"->"기기", "NANO"->"나노",
      "GAS"->"가스", "PROGRAM"->"프로그램", "DEVELOPMENT"->"개발", "EDUCATION"->"교육", "QUALITY"->"품질", "TEST"->"테스트", "BACTERIA"->"박테리아",
      "GENOME"->"유전체", "SEQUENCING"->"시퀀싱", "DOMAIN"->"도메인", "GENETICS"->"유전학", "TUMOR"->"종양", "PATTERN"->"패턴",
      "CHANNEL"->"채널", "RESEARCH"->"연구", "VIRUS"->"바이러스", "MEASUREMENT"->"측정"
    )

    val inputdata = sc.union(dir.map( d => {
      sc.wholeTextFiles(d + "*.txt").
        map(x => {
          val id = x._1.substring(x._1.lastIndexOf("/") + 1, x._1.lastIndexOf("."))
          val lines = x._2.split("\n")
          var words = Array[String]()
          try {
            words = lines.flatMap(l => {
              val w = vocabWords(preprocess(l))
              w.map(x => if (replaceMap.contains(x)) replaceMap(x) else x)
            })
          } catch {
            case e: Exception => {
              println(e); println(id); println(lines)
            }
          }
          (id, words)
        })
      }
    ))

    val stopWordsSet = Set("OF", "AND", "ET", "AL", "IN", "AS", "IS", "FROM", "OR", "UP", "THE", "FOR", "BASED",
      "CAN", "ARE", "USING", "THIS", "THAT", "근간한", "대한", "활용", "관련", "가능", "연구", "개발", "통한",
      "제시", "제공", "이용", "적용", "다양", "0", "의한", "ET", "중요", "TO", "BE", "BY", "WITH", "WILL", "ON", "부터", "및", "기존", "주요", "대부분",
      "위한", "이용한", "개발", "관한", "미치는", "기반의", "활용한", "규명", "새로운", "기반", "유래", "기술", "시스템", "따른", "II", "때", "여러", "에서",
      "방안", "기법", "최근", "경우", "위해", "첫째", "둘째", "셋째", "때문", "모두", "RELATED", "아니", "이미", "REAL", "에서", "분야", "진행", "수행",
      "현재", "국내", "세계", "단계", "사용", "기대", "과제", "결과", "방법", "필요", "과학", "기술", "수준", "학문", "산업", "다른", "가장", "하나",
      "통하", "가지", "연구자", "대해", "아직", "대상", "이들", "사이", "기여", "효과", "발전", "국가", "구축", "핵심", "목표", "시장", "확보", "원천",
      "경제", "경쟁력", "특허", "향상", "측면", "고려", "구축", "학적", "하지", "실제", "바탕", "예정", "보고", "생각", "인해", "확인",
      "분석", "도출", "종합", "파악", "세", "우리", "제안", "논문", "학술", "시도", "접근", "방향", "계획", "형태", "증가", "성과")

    stopWords = sc.broadcast(stopWordsSet).value
    stopDocs = sc.broadcast(scala.io.Source.fromFile("data/stopdocs.txt").getLines().toSet).value

    inputdata.filter(x => !stopDocs.contains(x._1))
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

  def getMatrixTFIDF(corpus: RDD[Array[String]]) = {
    import org.apache.spark.mllib.feature.{HashingTF, IDF}
    val hashingTF = new HashingTF() // Hashing을 이용해서 Term Frequency를 구함
    val tf = hashingTF.transform(corpus.map(c => c.toIndexedSeq))
    tf.cache()

    val idf = new IDF().fit(tf)
    val vocab = corpus.flatMap(x => x).distinct.map(x => (hashingTF.indexOf(x), x)).collect.toMap
    val tfidf = idf.transform(tf)

    (vocab.map(_.swap), tfidf.zipWithIndex.map(_.swap))
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

    words.filter(s => !s.forall(_.isDigit) && !stoppattern(s))
  }


  /*val pattern = "[－()\\-,()/?.\uDBC1\uDE5B￭\uF06C\uF09E▷\uDB80\uDEEF<\uDEEF\uF0A0１‘２\uF0D7･～\uF06D▶\uF09F\uDB80\uDEFB:" +
    "\uDB80\uDEEB\uD835\uDF70~=---\";;「ㆍ’'“”／·•⦁▸◎>○\uF061：　╸∎▪◦˚◼︎●■→*（(((，茶．·＜＞+①②③➌④" +
    "３□）)))九六補瀉法︎\uF02D\uDB80\uDEEE龍脈\uDB80\uDEB1\uDB80\uDEB2\uDB80\uDEB3捻轉ㅃ∙㉮ㅇ///◆\uD835\uDEC3＊\uDBFA\uDF16]" //
    */
  val pattern = "[^(가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9)]"

  // 변환. 영어는 모두 대분자로
  def preprocess(s: String) = {
    s.toUpperCase.replaceAll(pattern, " ").replaceAll("[()/:]", " ")//.replaceAll(pattern2, " ").replaceAll("]", " ")
  }

  def stoppattern(s: String): Boolean = {
    val pattern = "[0-9]+(차|개|년|시|세|점|번|명|위|년|차년|회|주|차원|곳|만명|개월|분|차년도|단계|가지|년차)"
    s.matches(pattern)
  }
}

trait TFIDF {
  def getMatrix(corpus: RDD[Array[String]]) : (RDD[Vector], HashingTF, IDFModel) = {
    val hashingTF = new HashingTF()
    val tf: RDD[Vector] = hashingTF.transform(corpus.map(c => c.toIndexedSeq))

    import org.apache.spark.mllib.feature.IDF
    tf.cache()
    val idf = new IDF().fit(tf)
    (idf.transform(tf), hashingTF, idf)
  }

  def getMatrixFreqOnly(corpus: RDD[Array[String]]) : (RDD[Vector], Map[String, Int]) = {
    val vocab = corpus.flatMap(x => x).distinct.collect.zipWithIndex.toMap
    val matrix: RDD[Vector] = corpus.map {
      case tokens => {
        //val counts = new scala.collection.mutable.HashMap[Int, Double]()
        val vec = tokens.foldLeft(Map[Int, Double]())((res, t) => {
          val vocabid = vocab(t)
          res + (vocabid -> (res.getOrElse(vocabid, 0.0) + 1.0))
        })
        //val (indices, values) = vec.keys
        new SparseVector(vocab.size, vec.keys.toArray, vec.values.toArray)
      }
    }
    (matrix, vocab)
  }
}

object PreprocessingTest extends PreProcessing {
  def main(args: Array[String]) : Unit = {
    println(preprocess("aaaaa한글ab-b123*b)ccc•⦁▸◎dddii) 통합 유무선네트워크(Integrated Ship Area Network: i-SAN) 및 W"))

    Array("1차", "2개", "4점", "3차전", "110개").foreach(s => println(s + ":" + stoppattern(s)))
  }
}
