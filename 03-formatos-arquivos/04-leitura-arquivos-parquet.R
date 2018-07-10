library(sparklyr)
sc <- spark_connect(master = "local")

library(tidyverse)
library(sparkavro)
library(sparklyr.nested)

ratings_music <- spark_read_parquet(sc, name = "ratings_music", path = "03-formatos-arquivos/data/subset_meta_Digital_Music.parquet", memory = FALSE)

ratings_music %>%
  select(asin, description, imUrl, price, title) %>%
  show_query()

ratings_music %>%
  select(asin, description, imUrl, price, title)
  
ratings_music %>%
  select(asin, description, imUrl, price, title) %>%
  filter(price > 15)

spark_disconnect_all()
