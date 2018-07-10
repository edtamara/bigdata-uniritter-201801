from pyspark.sql.functions import expr, col, column

spark.read.parquet("s3a://aula-spark/airlines_parquet")\
  .createOrReplaceTempView("some_sql_view")

spark.sql("""
SELECT year, month, day, dayofweek, dep_time, crs_dep_time, arr_time, crs_arr_time, cast( carrier AS STRING ), flight_num, tail_num
     , actual_elapsed_time, crs_elapsed_time, airtime, arrdelay, depdelay, cast( origin AS STRING ), cast( dest AS STRING ), distance
     , taxi_in, taxi_out, cancelled, cast( cancellation_code AS STRING ), diverted, carrier_delay, weather_delay, nas_delay
     , security_delay, late_aircraft_delay
  FROM some_sql_view
""").createOrReplaceTempView("flight_data")

flight_data = spark.table("flight_data")

flight_data.groupBy("year", "carrier", "origin", "dest").count().where(flight_data.year == 2007).where(flight_data.carrier == "AA").explain()

AA_2007_flights = flight_data.groupBy("year", "carrier", "origin", "dest").count().where((flight_data.year == 2007) & (flight_data.carrier == "AA"))



# Import dinâmico de funções do Spark SQL que são definidas em Scala!
from pyspark.sql.functions import sum, count, avg

summaries = flight_data\
   .groupBy("flight_num", "carrier", "origin", "dest")\
   .agg(
      count("year").alias("quan"),
      expr("COUNT(1) AS quan2"),
      avg("actual_elapsed_time"),
      avg("airtime")\
   )

spark.sql("select count(1) from flight_data where airtime IS NULL").show()

AA_2007_flights.write.mode("overwrite").csv("/tmp/flights_aa.csv")

AA_2007_flights.coalesce(1).write.mode("overwrite").csv("/tmp/flights_aa.csv")
