import pandas as pd

spark.read.parquet("s3a://aula-spark/airlines_parquet")\
  .createOrReplaceTempView("some_sql_view") # DF => SQL

spark.table("some_sql_view").printSchema()

# O que há com as colunas String??
spark.table("some_sql_view").select("dest").show(20)

spark.sql("""
SELECT year, month, day, dayofweek, dep_time, crs_dep_time, arr_time, crs_arr_time, cast( carrier AS STRING ), flight_num, tail_num
     , actual_elapsed_time, crs_elapsed_time, airtime, arrdelay, depdelay, cast( origin AS STRING ), cast( dest AS STRING ), distance
     , taxi_in, taxi_out, cancelled, cast( cancellation_code AS STRING ), diverted, carrier_delay, weather_delay, nas_delay
     , security_delay, late_aircraft_delay
  FROM some_sql_view
""").createOrReplaceTempView("flight_data")

destinos_s_1000_mais = spark.sql("""
SELECT dest
     , COUNT(1)
  FROM flight_data 
 GROUP BY dest
""")\
  .where("dest LIKE 'S%'").where("`COUNT(1)` > 1000")

# Quantas colunas são consultadas? Quantos casts são executados?
# Em que ordem ocorrem o filtro de like, o agrupamento, e o filtro de count?
destinos_s_1000_mais.explain()

aeroportos = spark.read.csv("s3a://aula-spark/airports.csv", header=True, inferSchema=True)

aeroportos.createOrReplaceTempView("airports")

# Como mudar a forma de referenciar o join e o alias?
# Tipo de Join executado
destinos = destinos_s_1000_mais\
    .join(aeroportos, aeroportos.iata == destinos_s_1000_mais.dest)\
    .select("dest", destinos_s_1000_mais["`COUNT(1)`"].alias("qtty"),  "airport", "city", "state")

destinos.explain()

destinos.show(20)

destinos.count()

pd_destinos = destinos.toPandas()

pd_destinos
