

name := "NRFAnalysis"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.0",
  "org.apache.spark" %% "spark-mllib" % "2.0.0",
  "joda-time" % "joda-time" % "2.7",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "breeze-natives" % "0.11.2",
  "org.scalanlp" %% "breeze-viz" % "0.11.2"
)

libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.0" % "test"

libraryDependencies += "org.apache.poi" % "poi" % "3.14"
libraryDependencies += "org.apache.poi" % "poi-ooxml" % "3.14"

// http://mvnrepository.com/artifact/org.apache.lucene/lucene-analyzers
libraryDependencies += "org.apache.lucene" % "lucene-analyzers" % "3.6.2"


libraryDependencies += "arirang.lucene-analyzer-5.0" % "arirang.lucene-analyzer-5.0" % "1.0.0" from "file:///Users/chanjinpark/GitHub/NRFAnalysis/jars/arirang.lucene-analyzer-5.0-1.0.0.jar"
libraryDependencies += "com.argo" % "arirang-morph" % "1.0.0" from "file:///Users/chanjinpark/GitHub/NRFAnalysis/jars/arirang-morph-1.0.0.jar"



//libraryDependencies += "org.apache.tika" % "tika-parsers" % "1.13"