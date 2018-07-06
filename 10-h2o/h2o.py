#ec2-23-20-137-129.compute-1.amazonaws.com
#cd sparkling-water-2.3.8
#PYSPARK_DRIVER_PYTHON="ipython" bin/pysparkling --conf spark.scheduler.minRegisteredResourcesRatio=1 --conf spark.dynamicAllocation.enabled=false --conf spark.executor.instances=5

from pysparkling import *

conf = H2OConf(spark)

hc = H2OContext.getOrCreate(spark, conf)

taxi_data = spark.read.parquet("s3a://aula-spark/yellow_tripdata_2017.parquet")

taxi_rate = taxi_data.where('payment_type = 1 AND trip_distance >= 1').selectExpr('tip_amount / fare_amount as tip_rate', 'passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tolls_amount', 'hour(tpep_pickup_datetime) as hour', "date_format(tpep_pickup_datetime, 'EEEE')  as weekday").where('COALESCE(tip_rate, 0) BETWEEN 0.001 AND 0.99')

h2o_taxi_rate = hc.as_h2o_frame(taxi_rate, "taxi_rate")

h2o_taxi_rate['weekday'] = h2o_taxi_rate['weekday'].asfactor()

train, test, valid = h2o_taxi_rate.split_frame(ratios=[.7, .15])


from h2o.estimators.glm import *

glm_classifier = H2OGeneralizedLinearEstimator(family="gaussian", nfolds=10, alpha=[0.05, 0.5, 0.95], lambda_search=True, nlambdas = 50, remove_collinear_columns = True)

glm_classifier.train( y="tip_rate", x=['passenger_count', 'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tolls_amount', 'hour', 'weekday'], training_frame = train )

glm_classifier._model_json['output']
