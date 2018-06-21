import pandas as pd
from pyspark.sql.functions import col

spark.read.parquet("s3a://aula-spark/airlines_parquet")\
  .createOrReplaceTempView("some_sql_view") # DF => SQL

spark.table("some_sql_view").groupBy("year").count().sort(col("year").desc()).toPandas()


spark.sql("SELECT year, count(1) AS count FROM some_sql_view GROUP BY year ORDER BY year DESC").show(30)
