WITH q_daily_avg_fare AS (
     SELECT AVG( fare_amount ) AS AVG_FARE
          , RatecodeID
          , DATE(tpep_pickup_datetime) AS PICKUP_DATE
       FROM nyc_parquet
      WHERE DATE(tpep_pickup_datetime) >= TO_DATE( '2017-01-01' )
      GROUP BY PICKUP_DATE
             , RatecodeID )
 SELECT RatecodeID
      , PICKUP_DATE
      , AVG_FARE
      , AVG_FARE - LAG(AVG_FARE, 1) OVER ( PARTITION BY RatecodeID ORDER BY PICKUP_DATE ) AS DIFF_PREVIOUS 
   FROM q_daily_avg_fare
  ORDER BY RatecodeID
         , PICKUP_DATE;
