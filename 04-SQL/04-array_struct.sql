WITH rides_from_pickup_location AS (
     SELECT pickup_location_id
          , TIMESTAMP_TRUNC( pickup_datetime, DAY ) AS PICKUP_DATE
          , ARRAY_AGG( 
               STRUCT( dropoff_location_id
                     , fare_amount
                     , rate_code
                     , trip_distance ) 
               ORDER BY trip_distance ASC ) AS destinations
       FROM `bigquery-public-data.new_york_taxi_trips.tlc_yellow_trips_2017`
      WHERE rate_code <> 99
      GROUP BY pickup_location_id
             , PICKUP_DATE
)

SELECT * 
  FROM rides_from_pickup_location
