### NSFAwards 분석 - SVD 코드
NSF Awards 데이터에 대해서 SVD 분석을 하고, IDF 기반 Term Document Matrix 구축 및 활용 

Lemmaize

```scala
package LDA

import java.io.{FileOutputStream, PrintStream}
import java.util.Properties

import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


import edu.stanford.nlp.ling.CoreAnnotations.{LemmaAnnotation, SentencesAnnotation, TokensAnnotation}
import edu.stanford.nlp.ling.CoreLabel
import edu.stanford.nlp.pipeline.{Annotation, StanfordCoreNLP}
import edu.stanford.nlp.util.CoreMap

import scala.collection.Map
import scala.collection.mutable.ArrayBuffer

/**
  * Created by chanjinpark on 2016. 4. 8..
  */
object NSFAwards {
  def main(args: Array[String]) = {
    val conf = new SparkConf(true).setMaster("local").setAppName("NSFLDA")
    val sc = new SparkContext(conf)

    Logger.getLogger("org").setLevel(Level.ERROR)

    // spark-shell --master local[4] --jars code.jar

    val path = "/Users/chanjinpark/DevProj/NSFAwards/data/awards_2003/"

    val k = 100
    val numTerms = 5000
    val sampleSize = 1.0

    val inputdata = sc.wholeTextFiles(path + "*.txt").
      map(x => {
        val kv = parseProject(x._2.split("\n").toList)
        (kv("File"), kv("Title"),  kv("Abstract"), kv("fields"))
      })

    val plainText = inputdata.map(d => (d._1 + ":" + d._4, d._2 + " " + d._3))
    
    val stopWords = {
      def loadStopWords(path: String) = scala.io.Source.fromFile(path).getLines().toSet
      sc.broadcast(loadStopWords("./stopwords.txt")).value
    }

    val lemmatized = plainText.mapPartitions(iter => {
      // createNLPPipeline
      val props = new Properties()
      props.put("annotators",  "tokenize, ssplit, pos, lemma") //"tokenize, ssplit, pos, lemma, ner, parse, dcoref"
      val pipeline = new StanfordCoreNLP(props)
      // lemmatize
      iter.map { case(title, contents) => (title, lemmatize(contents, stopWords, pipeline)) }
    }).filter(_._2.size > 1)
    
    // 추가 필터 사용
    /*filter(s => !("research,researcher,participant,community,university,provide,use,student,problem,new,workshop,develop,hold,international," +
                  "science,engineering,conference,area,engineering,group,study,analysis,program,project,work,system,undergraduate,technology,education," +
                  "support,propose,design,result,effect,year,proposal,include,development,understand,understanding,important").
                  split(",").map(_.trim).contains(s))
    */
    
    val (termDocMatrix, termIds, docIds, idfs) = termDocumentMatrix(lemmatized, stopWords, numTerms, sc)
    
    termDocMatrix.cache()
    
    val mat = new RowMatrix(termDocMatrix)
    val svd = mat.computeSVD(k, computeU=true)

    println("Singular values: " + svd.s)

    val topConceptTerms = topTermsInTopConcepts(svd, 10, 10, termIds)
    val topConceptDocs = topDocsInTopConcepts(svd, 10, 10, docIds)
    for ((terms, docs) <- topConceptTerms.zip(topConceptDocs)) {
      println("Concept terms: " + terms.map(_._1).mkString(", "))
      println("Concept docs: " + docs.map(_._1).mkString(", "))
      println()
    }
  }


  def parseProject(content: List[String]): Map[String, String] = {
    var kvlist = Map[String, String]()
    var (key, value) = ("", List[String]())
    content.foreach(l=> {
      if (l.contains(":")) {
        val pair = l.split(":").map(_.trim).toList
        if ( key.length > 0 ) 
          kvlist = kvlist + (key -> value.reverse.mkString(" "))
        
        key = pair.head
        value = pair.tail
      }
      else value = l.trim :: value
    })
    if (content.nonEmpty) 
      kvlist = kvlist + (key -> value.reverse.mkString(" "))

    var fields_ = List[String]()
    fields_ = fields_ ::: (if (kvlist.contains("NSF Program")) kvlist("NSF Program").split(",").toList else List[String]())
    fields_ = fields_ ::: (if (kvlist.contains("Fld Applictn")) kvlist("Fld Applictn").split(",").toList else List[String]())

    val fields = fields_.map(f => {
      if (f.length > 0 && Character.isDigit(f.charAt(0))) 
        f.substring(f.indexOf(" "), f.length).trim
      else f.trim
    }).filter(_.length > 0)
    kvlist + ("fields" -> fields.mkString(","))
  }
  
  import scala.collection.JavaConversions._
  def lemmatize(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP): Seq[String] = {
    val doc = new Annotation(text)
    pipeline.annotate(doc)
    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation]).toList

    sentences.foreach(sentence => {
      sentence.get(classOf[TokensAnnotation]).toList.foreach( token => {
        val lemma = token.get(classOf[LemmaAnnotation])
        if ( lemma.length > 2 && !stopWords.contains(lemma) && lemma.forall(java.lang.Character.isLetter)) { // && isOnlyLetters(lemma))
          lemmas += lemma.toLowerCase()
        }
      })
    })
    lemmas
  }

  def termDocumentMatrix(docs: RDD[(String, Seq[String])], stopWords: Set[String], numTerms: Int, sc: SparkContext)
    : (RDD[Vector], Map[Int, String], Map[Long, String], Map[String, Double]) = {

    val docTermFreqs = docs.mapValues(terms => {
      // term frequency in docs
      terms.foldLeft(new scala.collection.mutable.HashMap[String, Int]()) {
        (map, term) => map += term -> (map.getOrElse(term, 0) + 1) }
    })
    docTermFreqs.cache()
    
    val docIds = docTermFreqs.map(_._1).zipWithUniqueId().map(_.swap).collectAsMap()
    val docFreqs = documentFrequenciesDistributed(docTermFreqs.map(_._2), numTerms)
    saveDocFreqs("docfreqs.tsv", docFreqs)

    val numDocs = docIds.size
    val idfs = inverseDocumentFrequencies(docFreqs, numDocs)
    val termToId = idfs.keys.zipWithIndex.toMap // Maps terms to their indices in the vector

    val bIdfs = sc.broadcast(idfs).value
    val bTermToId = sc.broadcast(termToId).value
    val vecs = docTermFreqs.map(_._2).map(termFreqs => {
      val docTotalTerms = termFreqs.values.sum
      val termScores = termFreqs.filter {
        case (term, freq) => bTermToId.containsKey(term)
      }.map{
        case (term, freq) => (bTermToId(term), bIdfs(term) * termFreqs(term) / docTotalTerms)
      }.toSeq
      Vectors.sparse(bTermToId.size, termScores)
    })
    (vecs, termToId.map(_.swap), docIds, idfs)
  }

  def documentFrequenciesDistributed(docTermFreqs: RDD[scala.collection.mutable.HashMap[String, Int]], numTerms: Int)
  : Array[(String, Int)] = {
    val docFreqs = docTermFreqs.flatMap(_.keySet).map((_, 1)).reduceByKey(_ + _, 15)
    val ordering = Ordering.by[(String, Int), Int](_._2)
    docFreqs.top(numTerms)(ordering)
  }

  def saveDocFreqs(path: String, docFreqs: Array[(String, Int)]) {
    val ps = new PrintStream(new FileOutputStream(path))
    for ((doc, freq) <- docFreqs) { ps.println(s"$doc\t$freq") }
    ps.close()
  }

  def inverseDocumentFrequencies(docFreqs: Array[(String, Int)], numDocs: Int)
  : Map[String, Double] = {
    docFreqs.map{ case (term, count) => (term, math.log(numDocs.toDouble / count))}.toMap
  }
  
  def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                            numTerms: Int, termIds: Map[Int, String]): Seq[Seq[(String, Double)]] = {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(-_._1)
      topTerms += sorted.take(numTerms).map{case (score, id) => (termIds(id), score)}
    }
    topTerms
  }

  def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int,
                           numDocs: Int, docIds: Map[Long, String]): Seq[Seq[(String, Double)]] = {
    val u  = svd.U
    val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId
      topDocs += docWeights.top(numDocs).map{case (score, id) => (docIds(id), score)}
    }
    topDocs
  }
  
  def row(mat: BDenseMatrix[Double], index: Int): Seq[Double] = 
    (0 until mat.cols).map(c => mat(index, c))

  /* Selects a row from a matrix.*/
  def row(mat: Matrix, index: Int): Seq[Double] = {
    val arr = mat.toArray
    (0 until mat.numCols).map(i => arr(index + i * mat.numRows))
  }

  /* Selects a row from a distributed matrix. */
  def row(mat: RowMatrix, id: Long): Array[Double] = {
    mat.rows.zipWithUniqueId.map(_.swap).lookup(id).head.toArray
  }

  /* Finds the product of a dense matrix and a diagonal matrix represented by a vector.
     Breeze doesn't support efficient diagonal representations, so multiply manually. */
  def multiplyByDiagonalMatrix(mat: Matrix, diag: Vector): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs{case ((r, c), v) => v * sArr(c)}
  }

  /* Finds the product of a distributed matrix and a diagonal matrix represented by a vector.*/
  def multiplyByDiagonalMatrix(mat: RowMatrix, diag: Vector): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map(vec => {
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    }))
  }

  /* Returns a matrix where each row is divided by its length. */
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).map(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  /* Returns a distributed matrix where each row is divided by its length. */
  def rowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map(vec => {
      val length = math.sqrt(vec.toArray.map(x => x * x).sum)
      Vectors.dense(vec.toArray.map(_ / length))
    }))
  }

  /* Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest
     relevance scores to the given term.*/
  def topTermsForTerm(normalizedVS: BDenseMatrix[Double], termId: Int): Seq[(Double, Int)] = {
    // Look up the row in VS corresponding to the given term ID.
    val termRowVec = new BDenseVector[Double](row(normalizedVS, termId).toArray)

    // Compute scores against every term
    val termScores = (normalizedVS * termRowVec).toArray.zipWithIndex

    // Find the terms with the highest scores
    termScores.sortBy(-_._1).take(10)
  }

  /* Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest
     relevance scores to the given doc.*/
  def topDocsForDoc(normalizedUS: RowMatrix, docId: Long): Seq[(Double, Long)] = {
    // Look up the row in US corresponding to the given doc ID.
    val docRowArr = row(normalizedUS, docId)
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    // Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  /* Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest
     relevance scores to the given term. */
  def topDocsForTerm(US: RowMatrix, V: Matrix, termId: Int): Seq[(Double, Long)] = {
    val termRowArr = row(V, termId).toArray
    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def termsToQueryVector(terms: Seq[String], idTerms: Map[String, Int], idfs: Map[String, Double])
  : BSparseVector[Double] = {
    val indices = terms.map(idTerms(_)).toArray
    val values = terms.map(idfs(_)).toArray
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  def topDocsForTermQuery(US: RowMatrix, V: Matrix, query: BSparseVector[Double])
  : Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def printTopTermsForTerm(normalizedVS: BDenseMatrix[Double],
                           term: String, idTerms: Map[String, Int], termIds: Map[Int, String]) {
    printIdWeights(topTermsForTerm(normalizedVS, idTerms(term)), termIds)
  }

  def printTopDocsForDoc(normalizedUS: RowMatrix, doc: String, idDocs: Map[String, Long],
                         docIds: Map[Long, String]) {
    printIdWeights(topDocsForDoc(normalizedUS, idDocs(doc)), docIds)
  }

  def printTopDocsForTerm(US: RowMatrix, V: Matrix, term: String, idTerms: Map[String, Int],
                          docIds: Map[Long, String]) {
    printIdWeights(topDocsForTerm(US, V, idTerms(term)), docIds)
  }

  def printIdWeights[T](idWeights: Seq[(Double, T)], entityIds: Map[T, String]) {
    println(idWeights.map{case (score, id) => (entityIds(id), score)}.mkString(", "))
  }
}

```