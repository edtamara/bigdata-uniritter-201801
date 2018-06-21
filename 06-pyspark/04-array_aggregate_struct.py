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

from pyspark.sql.functions import collect_list, struct

flights_origins = flight_data\
   .where(col("year") == 2007)\
   .select("dest", "origin", "carrier", "distance")\
   .distinct()\
   .sort(col("distance"), col("carrier").desc())\
   .groupBy("dest").agg(collect_list( struct("origin", "carrier", "distance") ).alias("list_origins"))\
   .sort("dest")

flights_origins.explain()

flights_origins.show(50)

aeroportos = spark.read.csv("s3a://aula-spark/airports.csv", header=True, inferSchema=True).cache()

aeroportos.createOrReplaceTempView("airports")

flights_origins.createOrReplaceTempView("flights_origins")

aeroportos_voos_origem = spark.sql("SELECT dest, airport, city, list_origins FROM flights_origins INNER JOIN airports ON dest = iata")

# Ver plano de execução... cache, etc

# Em que ordem o resultado de aeroportos_voos_origem será exibido?

