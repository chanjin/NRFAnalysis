/**
  * Created by chanjinpark on 2016. 9. 7..
  */

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.mllib.linalg.{Vector, SparseVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


val sentenceData = sc.parallelize(Seq(
  (0.0, "Hi I heard about Spark"),
  (0.0, "I wish Java could use case classes"),
  (1.0, "Logistic regression models are neat")
))

val corpus = sentenceData.map(_._2.split(" "))

val (tfidf, hashtf) = {
  import org.apache.spark.mllib.feature.{HashingTF, IDF}
  val hashingTF = new HashingTF() // Hashing을 이용해서 Term Frequency를 구함
  val tf = hashingTF.transform(corpus.map(c => c.toIndexedSeq))
  tf.cache()

  val idf = new IDF(minDocFreq = 2).fit(tf)
  (idf.transform(tf), hashingTF)
}

val vocab = corpus.flatMap(x => x).distinct.map(x => (hashtf.indexOf(x), x)).collect.toMap
val tfidf1 = tfidf.map(v => (v.asInstanceOf[SparseVector].indices.zip(v.asInstanceOf[SparseVector].values).map(x => (vocab(x._1), x._2))))
tfidf1.foreach(x => println(x.mkString(", ")))

hashtf.indexOf("Hi")