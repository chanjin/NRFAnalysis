package basic.hangeul

/**
  * Created by chanjinpark on 2016. 6. 17..
  */


import org.apache.lucene.analysis._
import org.apache.lucene.analysis.ko._
import org.apache.lucene.analysis.ko.morph._

import scala.collection.JavaConversions._

object Test {

  def main(args: Array[String]): Unit = {

    val in1 = Array("올해 크리스마스에는 눈이 내리지 않고 비교적 포근할 - 전망이다. Piki는", "하늘공원 Test임.")
    val words = in1.flatMap(l =>
      HangeulAnalysis.vocabWords(HangeulAnalysis.preprocess(l))
    )
    println(words.mkString(","))
  }
}

object HangeulAnalysis {

  def main(args: Array[String]): Unit = {
    val in1 = "올해 크리스마스에는 눈이 내리지 않고 비교적 포근할 전망이다. Piki는"
    println(morphAnalyze(in1))
    println("-----------")

    val in2 = "하늘공원 Piki는"
    println(compoundNounAnalyze(in2))
    println

    val in3 = "올 해크리스마스 에는 눈이내리지않고 비교적포근할전 망이다. Piki는"
    //println(wordSpaceAnalyze(in3))
    //println

    val in4 = "올해 크리스마스에는 눈이 내리지 않고 비교적 포근할 전망이다 Test임."
    println(guideWords(in4))

    /*
    val workspace ="/Users/chanjinpark/data/NRF2015/"
    val contdir = workspace + "content/"


    val contsample = contdir + "2015R1D1A1A01058188" + ".txt" //2015R1D1A1A01058188


    scala.io.Source.fromFile(contsample).getLines().foreach(l => {
      println("<" + l + ">")
      println(guideWords(preprocess(l)))
      println()
    })

    val words = scala.io.Source.fromFile(contsample).getLines().flatMap(l =>
      vocabWords(preprocess(l))
    ).toList
    println(words.mkString(","))
    */
  }

  val pattern = "[^(가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9)]"
  def preprocess(s: String) = {
    s.replaceAll(pattern, " ").replaceAll("[()]", " ")//.replaceAll(pattern2, " ").replaceAll("]", " ")
  }

  def morphAnalyze(source: String) = {
    val ma = new MorphAnalyzer
    source.split(" ").map(x => {
      val res = ma.analyze(x)
      println(res.mkString("\t") + ":" + res.map(_.getPos).mkString("\t"))
      res.filter(_.getPos == PatternConstants.POS_NOUN).map(r => r.getStem).mkString("\t")
    }).mkString("\n")
  }

  def nounExtract(s: String) ={

  }

  def compoundNounAnalyze(source: String) = {
    val ca = new CompoundNounAnalyzer
    val out = ca.analyze(source).map(_.asInstanceOf[CompoundEntry]).toList
    out.map(_.getWord).mkString(" ")
  }

  def guideWords(source: String) = {
    val ma = new MorphAnalyzer
    source.split(" ").filter(_.length > 1).map(s => {
      println(s)
      val morphemes = ma.analyze(s).map(_.asInstanceOf[AnalysisOutput])

      val o = morphemes.filter(m => m.getPos == 'N')
      s + " - " + o.map(x => x.getStem + " - (" + x.getCNounList.map(_.getWord).mkString(" ") + ")").mkString("\n") +
        "\n\t==" + morphemes.filter(_.getPos != 'N').mkString("==")
    }).mkString("\n")
  }

  def vocabWords(source: String): Array[String] = {
    val ma = new MorphAnalyzer
    source.split(" ").filter(_.length > 1).flatMap(s => {
      println(s)
      val morphemes = ma.analyze(s.trim).map(_.asInstanceOf[AnalysisOutput])
      val o = morphemes.filter(m => m.getPos == 'N')

      print("- start " + s + ", " + o.length + ": ")
      o.map(_.getStem).filter(t => t.length > 1).mkString(",")foreach(print)
      println(" - end")

      //val stems = o.map(_.getStem).filter(t => t.length > 1)

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
  }
}
