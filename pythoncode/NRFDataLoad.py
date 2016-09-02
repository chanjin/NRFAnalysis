import os
import sys

print ("start")
os.environ['SPARK_HOME'] = "/Users/chanjinpark/dev/spark-1.6.2/"
sys.path.append("/Users/chanjinpark/dev/spark-1.6.2/python")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
except ImportError as e:
    print("Can't import pyspark ", e)
    sys.exit(1)

