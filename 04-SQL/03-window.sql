WITH q_daily_avg_fare AS (
     SELECT AVG( fare_amount ) AS AVG_FARE
          , rate_code
          , TIMESTAMP_TRUNC( pickup_datetime, DAY ) AS PICKUP_DATE
       FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
      WHERE pickup_datetime >= '2017-01-01'
      GROUP BY PICKUP_DATE
             , rate_code )

 SELECT rate_code
      , PICKUP_DATE
      , AVG_FARE
      , AVG_FARE - LAG(AVG_FARE, 1) OVER ( PARTITION BY rate_code ORDER BY PICKUP_DATE ) AS DIFF_PREVIOUS 
   FROM q_daily_avg_fare
  ORDER BY rate_code
         , PICKUP_DATE;
