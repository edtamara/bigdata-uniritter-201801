WITH q_trip_type AS (
      SELECT rate_code
           , trip_distance
           , fare_amount
           , IF( trip_distance < 1, "short", "long" ) trip_distance_type
        FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017` )

   , q_rate AS (
      SELECT rate_code
           , AVG( fare_amount / IF( trip_distance_type = "short", 1, trip_distance )) as avg_fare_amount_per_mile
        FROM q_trip_type
       GROUP BY rate_code )

SELECT trip_distance_type
     , y.rate_code
     , count(*)
  FROM q_trip_type AS y
 INNER JOIN q_rate AS q
         ON y.rate_code = q.rate_code
 WHERE y.fare_amount < q.avg_fare_amount_per_mile
 GROUP BY trip_distance_type
        , y.rate_code
 ORDER BY y.rate_code
        , trip_distance_type;