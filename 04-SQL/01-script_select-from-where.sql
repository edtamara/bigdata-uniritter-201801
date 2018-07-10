SELECT passenger_count
     , round(trip_distance) as rounded_distance
     , count(*) as trips_this_year
  FROM [bigquery-public-data:new_york_taxi_trips.tlc_yellow_trips_2017]
 GROUP BY passenger_count
        , rounded_distance
HAVING trips_this_year > 1
 ORDER BY trips_this_year DESC
 LIMIT 5;
