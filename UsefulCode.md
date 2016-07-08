
### Use mapPartition to skip csv's header

```scala
    val month = sc.textFile(workspacePath + s"totdata-$i").mapPartitionsWithIndex(
      (i, iterator) => if (i == 0 && iterator.hasNext) {
        iterator.next;
        iterator
      } else iterator)
    month.map(s => s.split(",")).map(arr => (arr(1).toInt, arr(2).toInt))
```