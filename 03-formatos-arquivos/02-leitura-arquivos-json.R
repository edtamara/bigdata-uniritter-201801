library(sparklyr)
sc <- spark_connect(master = "local")

library(tidyverse)

ratings_music <- spark_read_json(sc, name = "ratings_music", path = "03-formatos-arquivos/data/subset_meta_Digital_Music.json.gz", memory = FALSE)
# ratings_music_ <- spark_read_json(sc, name = "ratings_music_", path = "03-formatos-arquivos/data/pretty_Digital_Music.json.gz", memory = FALSE)

sdf_schema(ratings_music)

# sdf_schema(ratings_music_)
# DBI::dbRemoveTable(sc, "ratings_music_")

head(ratings_music)

ratings_music %>%
  filter(asin == "6308051551") %>%
  collect() %>%
  View()

# Quais são as 100 marcas (brands) mais comuns? Das 100 mais comuns, exiba todas aquelas com no mínimo 2 ocorrências e mostre na console a query resultante desta análise
ratings_music %>%
  count(brand) %>%
  arrange(desc(n)) %>%
  head(100) %>%
  filter(n > 1) %>%
  collect() %>%
  print(n = 100)

ratings_music %>%
  count(brand) %>%
  arrange(desc(n)) %>%
  head(100) %>%
  filter(n > 1) %>% show_query()

#devtools::install_github("chezou/sparkavro")
#copiar jar databricks avro para as libs da cópia local do spark (ver local no sc)
#devtools::install_github("mitre/sparklyr.nested")

library(sparkavro)

spark_write_avro(ratings_music, path = "03-formatos-arquivos/data/subset_meta_Digital_Music.avro", options = list(compression = "snappy"))

spark_disconnect(sc)
spark_disconnect_all()
