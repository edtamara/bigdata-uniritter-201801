val nyc_parquet = spark.read.parquet("s3a://aula-spark/yellow_tripdata_2017.parquet")

nyc_parquet.printSchema()

nyc_parquet.select("payment_type").distinct().explain()
nyc_parquet.select("payment_type").distinct().show()

nyc_parquet.groupBy("payment_type").count().explain()
nyc_parquet.groupBy("payment_type").count().show()

nyc_parquet.groupBy("payment_type").count().orderBy(col("count").desc).show()

nyc_parquet.where("trip_distance <= 1").groupBy("payment_type").count().orderBy(col("count").desc).explain()
nyc_parquet.where("trip_distance <= 1").groupBy("payment_type").count().orderBy(col("count").desc).show()
