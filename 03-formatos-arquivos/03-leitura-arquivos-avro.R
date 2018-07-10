library(sparklyr)
sc <- spark_connect(master = "local")

library(tidyverse)
library(sparkavro)
library(sparklyr.nested)

ratings_music <- spark_read_avro(sc, name = "ratings_music", path = "03-formatos-arquivos/data/subset_meta_Digital_Music.avro", memory = FALSE)

sdf_schema(ratings_music)

# java -jar ./avro-tools-1.7.7.jar getschema ~/Documents/Uniritter/2018/big-data/03-formatos-arquivos/data/subset_meta_Digital_Music.avro/*.avro
# java -jar ./avro-tools-1.7.7.jar tojson --pretty ~/Documents/Uniritter/2018/big-data/03-formatos-arquivos/data/subset_meta_Digital_Music.avro/*.avro | head -1000

ratings_music %>%
  sdf_select(asin, related$bought_together) %>%
  sdf_explode(bought_together) %>%
  arrange(asin)

ratings_music %>%
  sdf_select(asin, related$bought_together) %>% 
  sdf_explode(bought_together) %>%
  arrange(asin) %>%
  count(asin) %>%
  arrange(desc(n))

ratings_music %>%
  select(title)

spark_write_parquet(ratings_music, path = "03-formatos-arquivos/data/subset_meta_Digital_Music.parquet")

spark_disconnect(sc)
