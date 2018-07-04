import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.log4j.LogManager

Logger.getLogger("org").setLevel(Level.OFF)
Logger.getLogger("akka").setLevel(Level.OFF)

LogManager.shutdown()
LogManager.resetConfiguration()
LogManager.getLogger("MdsLogger").setLevel(Level.OFF)

val order_products = spark.read.format("csv").option("header", true).option("inferSchema", true).load("instacart/order_products*.csv").repartition( 6 )
val products = spark.read.format("csv").option("header", true).option("inferSchema", true).load("instacart/product*.csv").coalesce( 1 )

order_products.write.parquet("instacart_binary/order_products")
products.write.parquet("instacart_binary/products")

val cart = order_products.select(col("order_id"), col("product_id")).join( products.select( col("product_id").as("prd_id"), col("product_name") ), col("product_id") === col("prd_id"), "inner" )

val cart_array = cart.groupBy("order_id").agg(collect_list( "product_name" ).alias("items")).coalesce( 5 )

import org.apache.spark.ml.fpm.FPGrowth

val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(0.001).setMinConfidence(0.15)
val model = fpgrowth.fit(cart_array)

// Mosta os conjuntos frequentes
model.freqItemsets.show()

// Mostra as regras de associação geradas
model.associationRules.show()

// Gera predições de próximo item
model.transform(cart_array).show()

