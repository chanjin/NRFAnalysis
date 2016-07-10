

name := "NRFAnalysis"

version := "1.0"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.6.2",
  "org.apache.spark" % "spark-mllib_2.10" % "1.6.2",
  "joda-time" % "joda-time" % "2.7",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2"
)


libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0"
libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models"


libraryDependencies += "org.apache.poi" % "poi" % "3.14"
libraryDependencies += "org.apache.poi" % "poi-ooxml" % "3.14"

// http://mvnrepository.com/artifact/org.apache.lucene/lucene-analyzers
libraryDependencies += "org.apache.lucene" % "lucene-analyzers" % "3.6.2"


//libraryDependencies += "org.apache.tika" % "tika-app" % "1.13"

//libraryDependencies += "org.apache.tika" % "tika-core" % "1.13"

//libraryDependencies += "org.apache.tika" % "tika-parsers" % "1.13"