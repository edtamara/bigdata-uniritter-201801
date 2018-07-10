val nyc_jan = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("s3a://nyc-tlc/trip data/yellow_tripdata_2017-01.csv")
nyc_jan.summary().show()

nyc_jan.printSchema()

nyc_jan.summary.write.parquet( "s3a://aula-spark/class_yellow_tripdata_2017-01.parquet" )
