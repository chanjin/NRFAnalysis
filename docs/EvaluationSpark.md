### 분류 모델에 대한 평가 (Classification Model Evaluation)
[Spark Documentation URL](https://spark.apache.org/docs/latest/mllib-evaluation-metrics.html)

[분류에 대한 평가 방법 개요](#Binary)
[멀티 클래스 분류 평가](#multiclass)
[멀티 레이블 분류 평가](#multilabel)
[랭킹 평가](#ranking)
[회귀 평가](#regression)

#### 분류는 지도형 학습 모델 <a name="Binary"></a>
* 실제값과 예측값에 대해서 4가지 범주로 구분
    * True Positive (TP) - label is positive and prediction is also positive
    * True Negative (TN) - label is negative and prediction is also negative
    * False Positive (FP) - label is negative but prediction is positive
    * False Negative (FN) - label is positive but prediction is negative
    * (예측이 맞았나 틀렸나)(예측이 긍정/부정)
* 예측의 맞고 틀림 (Accuracy = TP + TN)은 좋은 메트릭이 아님
    * 데이터 셋이 Unbalanced 되어 있을 수 있음. 찾고자 하는 것에 100에 1이면, 1을 찾는게 중요. 99를 예측하는 것은 쉬움
    * 5%의 사기탐지 문제에서, 사기가 아닌 95%를 맞추는 것은 의미가 없음
    * 그래서, Precision과 Recall을 같이 보는 것이 중요. F-measure를 사용하기도 함
    * TP에 대한 상대적인 값. Precision은 긍정으로 예측한 것 중 TP, Recall은 참인 전체 중 TP.
    * ROC, AUC도 사용. ROC - Receiver Operating Characteristic, Area Under ROC Curve, Area Under Precision-Recall Curve

```scala
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

// Load training data in LIBSVM format
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_binary_classification_data.txt")

// Split data into training (60%) and test (40%)
val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
training.cache()

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(2)
  .run(training)

// Clear the prediction threshold so the model will return probabilities
model.clearThreshold

// Compute raw scores on the test set
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Instantiate metrics object
val metrics = new BinaryClassificationMetrics(predictionAndLabels)

// Precision by threshold
val precision = metrics.precisionByThreshold
precision.foreach { case (t, p) =>
  println(s"Threshold: $t, Precision: $p")
}

// Recall by threshold
val recall = metrics.recallByThreshold
recall.foreach { case (t, r) =>
  println(s"Threshold: $t, Recall: $r")
}

// Precision-Recall Curve
val PRC = metrics.pr

// F-measure
val f1Score = metrics.fMeasureByThreshold
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 1")
}

val beta = 0.5
val fScore = metrics.fMeasureByThreshold(beta)
f1Score.foreach { case (t, f) =>
  println(s"Threshold: $t, F-score: $f, Beta = 0.5")
}

// AUPRC
val auPRC = metrics.areaUnderPR
println("Area under precision-recall curve = " + auPRC)

// Compute thresholds used in ROC and PR curves
val thresholds = precision.map(_._1)

// ROC Curve
val roc = metrics.roc

// AUROC
val auROC = metrics.areaUnderROC
println("Area under ROC = " + auROC)
```
    
#### 멀티 클래스 분류 문제 <a name="multiclass"></a>
예측이 맞았다/틀렸다는 해당 분류에 속하냐 아니냐의 문제
예측값은 한 클래스에 긍정이고 다른 모든 클래스에 대해서는 부정임
주어진 데이터에 대해 True Negative가 많아짐

* 메트릭
    * Confusion Matrix
    * Accuracy, Precision by label, Recall by label, F-measure by label
    * Weighted precision, Weighted recall, Weighted F-measure
    
```scala
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

// Load training data in LIBSVM format
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_multiclass_classification_data.txt")

// Split data into training (60%) and test (40%)
val Array(training, test) = data.randomSplit(Array(0.6, 0.4), seed = 11L)
training.cache()

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS()
  .setNumClasses(3)
  .run(training)

// Compute raw scores on the test set
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

// Instantiate metrics object
val metrics = new MulticlassMetrics(predictionAndLabels)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Overall Statistics
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}

// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}

// False positive rate by label
labels.foreach { l =>
  println(s"FPR($l) = " + metrics.falsePositiveRate(l))
}

// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}

// Weighted stats
println(s"Weighted precision: ${metrics.weightedPrecision}")
println(s"Weighted recall: ${metrics.weightedRecall}")
println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
```


#### 멀티 레이블 분류 평가 <a name="multilable"></a>
한 데이터 샘플에 대해 여러 개 Label이 부여되는 경우 예를 들어 Tagging
뉴스의 토픽 
예측 결과에 있는 클래스 레이블이 실제 값에도 있는 경우 True
when that class exists in the predicted set and it exists in the true label set, for a specific data point

Precision, Recall, Accuracy
Precision by label, Recall by label, F1-measure by label
Hamming loss, Subset Accuracy, F1 measure 
Micro precision, Micro recall, Micro F1 Measure

```scala
import org.apache.spark.mllib.evaluation.MultilabelMetrics
import org.apache.spark.rdd.RDD

val scoreAndLabels: RDD[(Array[Double], Array[Double])] = sc.parallelize(
  Seq((Array(0.0, 1.0), Array(0.0, 2.0)),
    (Array(0.0, 2.0), Array(0.0, 1.0)),
    (Array.empty[Double], Array(0.0)),
    (Array(2.0), Array(2.0)),
    (Array(2.0, 0.0), Array(2.0, 0.0)),
    (Array(0.0, 1.0, 2.0), Array(0.0, 1.0)),
    (Array(1.0), Array(1.0, 2.0))), 2)

// Instantiate metrics object
val metrics = new MultilabelMetrics(scoreAndLabels)

// Summary stats
println(s"Recall = ${metrics.recall}")
println(s"Precision = ${metrics.precision}")
println(s"F1 measure = ${metrics.f1Measure}")
println(s"Accuracy = ${metrics.accuracy}")

// Individual label stats
metrics.labels.foreach(label =>
  println(s"Class $label precision = ${metrics.precision(label)}"))
metrics.labels.foreach(label => println(s"Class $label recall = ${metrics.recall(label)}"))
metrics.labels.foreach(label => println(s"Class $label F1-score = ${metrics.f1Measure(label)}"))

// Micro stats
println(s"Micro recall = ${metrics.microRecall}")
println(s"Micro precision = ${metrics.microPrecision}")
println(s"Micro F1 measure = ${metrics.microF1Measure}")

// Hamming loss
println(s"Hamming loss = ${metrics.hammingLoss}")

// Subset accuracy
println(s"Subset accuracy = ${metrics.subsetAccuracy}")
```

#### Ranking Systems <a name="ranking"></a>
추천이나 랭킹 시스템에서 사용

Precision at k
Mean Average Precision
Normalized Discounted Cumulative Gain

```scala
import org.apache.spark.mllib.evaluation.{RankingMetrics, RegressionMetrics}
import org.apache.spark.mllib.recommendation.{ALS, Rating}

// Read in the ratings data
val ratings = spark.read.textFile("data/mllib/sample_movielens_data.txt").rdd.map { line =>
  val fields = line.split("::")
  Rating(fields(0).toInt, fields(1).toInt, fields(2).toDouble - 2.5)
}.cache()

// Map ratings to 1 or 0, 1 indicating a movie that should be recommended
val binarizedRatings = ratings.map(r => Rating(r.user, r.product,
  if (r.rating > 0) 1.0 else 0.0)).cache()

// Summarize ratings
val numRatings = ratings.count()
val numUsers = ratings.map(_.user).distinct().count()
val numMovies = ratings.map(_.product).distinct().count()
println(s"Got $numRatings ratings from $numUsers users on $numMovies movies.")

// Build the model
val numIterations = 10
val rank = 10
val lambda = 0.01
val model = ALS.train(ratings, rank, numIterations, lambda)

// Define a function to scale ratings from 0 to 1
def scaledRating(r: Rating): Rating = {
  val scaledRating = math.max(math.min(r.rating, 1.0), 0.0)
  Rating(r.user, r.product, scaledRating)
}

// Get sorted top ten predictions for each user and then scale from [0, 1]
val userRecommended = model.recommendProductsForUsers(10).map { case (user, recs) =>
  (user, recs.map(scaledRating))
}

// Assume that any movie a user rated 3 or higher (which maps to a 1) is a relevant document
// Compare with top ten most relevant documents
val userMovies = binarizedRatings.groupBy(_.user)
val relevantDocuments = userMovies.join(userRecommended).map { case (user, (actual,
predictions)) =>
  (predictions.map(_.product), actual.filter(_.rating > 0.0).map(_.product).toArray)
}

// Instantiate metrics object
val metrics = new RankingMetrics(relevantDocuments)

// Precision at K
Array(1, 3, 5).foreach { k =>
  println(s"Precision at $k = ${metrics.precisionAt(k)}")
}

// Mean average precision
println(s"Mean average precision = ${metrics.meanAveragePrecision}")

// Normalized discounted cumulative gain
Array(1, 3, 5).foreach { k =>
  println(s"NDCG at $k = ${metrics.ndcgAt(k)}")
}

// Get predictions for each data point
val allPredictions = model.predict(ratings.map(r => (r.user, r.product))).map(r => ((r.user,
  r.product), r.rating))
val allRatings = ratings.map(r => ((r.user, r.product), r.rating))
val predictionsAndLabels = allPredictions.join(allRatings).map { case ((user, product),
(predicted, actual)) =>
  (predicted, actual)
}

// Get the RMSE using regression metrics
val regressionMetrics = new RegressionMetrics(predictionsAndLabels)
println(s"RMSE = ${regressionMetrics.rootMeanSquaredError}")

// R-squared
println(s"R-squared = ${regressionMetrics.r2}")
```

#### 회귀 모델 평가 <a name="regression"></a>

MSE - Mean Squared Error
RMSE - Root Mean Squared Error
MAE - Mean Absolute Error or Percentage Error
R square - Coefficient of Determination
EV - Explained Variance

```scala
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}

// Load the data
val data = spark
  .read.format("libsvm").load("data/mllib/sample_linear_regression_data.txt")
  .rdd.map(row => LabeledPoint(row.getDouble(0), row.get(1).asInstanceOf[Vector]))
  .cache()

// Build the model
val numIterations = 100
val model = LinearRegressionWithSGD.train(data, numIterations)

// Get predictions
val valuesAndPreds = data.map{ point =>
  val prediction = model.predict(point.features)
  (prediction, point.label)
}

// Instantiate metrics object
val metrics = new RegressionMetrics(valuesAndPreds)

// Squared error
println(s"MSE = ${metrics.meanSquaredError}")
println(s"RMSE = ${metrics.rootMeanSquaredError}")

// R-squared
println(s"R-squared = ${metrics.r2}")

// Mean absolute error
println(s"MAE = ${metrics.meanAbsoluteError}")

// Explained variance
println(s"Explained variance = ${metrics.explainedVariance}")
```