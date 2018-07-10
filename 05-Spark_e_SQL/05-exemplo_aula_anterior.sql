/*
 CREATE TEMPORARY VIEW nyc_parquet
  USING org.apache.spark.sql.parquet
OPTIONS (
          path "s3a://aula-spark/yellow_tripdata_2017.parquet"
);
*/

EXPLAIN WITH q_trip_type AS (
           SELECT RatecodeID
                , trip_distance
                , fare_amount
                , IF( trip_distance < 1, "short", "long" ) trip_distance_type
             FROM nyc_parquet )
           , q_rate AS (
           SELECT RatecodeID
                , AVG( fare_amount / IF( trip_distance_type = "short", 1, trip_distance )) AS avg_fare_amount_per_mile
             FROM q_trip_type
            GROUP BY RatecodeID )
 SELECT trip_distance_type
      , y.RatecodeID
      , count(*)
   FROM q_trip_type AS y
  INNER JOIN q_rate AS q
          ON y.RatecodeID = q.RatecodeID
  WHERE y.fare_amount < q.avg_fare_amount_per_mile
  GROUP BY trip_distance_type
         , y.RatecodeID
  ORDER BY y.RatecodeID
         , trip_distance_type;
