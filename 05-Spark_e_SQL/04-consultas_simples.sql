 CREATE TEMPORARY VIEW nyc_parquet
  USING org.apache.spark.sql.parquet
OPTIONS (
          path "s3a://aula-spark/yellow_tripdata_2017.parquet"
);

SELECT payment_type
     , COUNT( * ) AS count 
  FROM nyc_parquet 
 WHERE trip_distance <= 1 
 GROUP BY payment_type 
 ORDER BY count DESC;

EXPLAIN SELECT payment_type
             , COUNT( * ) AS count
          FROM nyc_parquet 
         WHERE trip_distance <= 1 
         GROUP BY payment_type 
         ORDER BY count DESC;
