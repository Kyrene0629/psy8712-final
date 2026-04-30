# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(tidyverse)
library(tidytext)
library(textstem)
library(tm)
library(stm)
library(Matrix)
library(caret)
library(glmnet)
library(ranger)
library(xgboost)
library(httr)
library(jsonlite)
library(parallel)
library(doParallel)
set.seed(1234)

glassdoor_raw_tbl <- read_csv("../data/glassdoor_reviews.csv")

ml_text_tbl <- glassdoor_raw_tbl %>%
  select(overall_rating, headline, pros, cons) %>%
  mutate(review_id = row_number(),
         review_text = str_squish(str_c(
           replace_na(headline, ""),
           replace_na(pros, ""),
           replace_na(cons, ""),
           sep = " "
         ))
         ) 
ml_text_tbl$review_text
nrow(ml_text_tbl)
# 838566

sample_n <- 10000
sample_prop <- sample_n / nrow(ml_text_tbl)

sample_text_tbl <- ml_text_tbl %>%
  group_by(overall_rating) %>%
  slice_sample(prop = sample_prop) %>%
  ungroup() %>%
  mutate(doc_id = row_number()) %>%
  select(doc_id, overall_rating, review_text)

# check the sample size and rating distribution
nrow(sample_text_tbl)
# 9998

sample_text_tbl %>%
  count(overall_rating) %>%
  mutate(prop = n / sum(n))

# Embeddings
get_embedding_batch <- function(text_vec) {
  response <- POST(
    url = "http://localhost:11434/api/embed",
    content_type_json(),
    body = list(
      model = "nomic-embed-text",
      input = text_vec
    ),
    encode = "json"
  )
  
  stop_for_status(response)
  
  result <- content(response, as = "parsed")
  embedding_mat <- do.call(rbind, result$embeddings)
  storage.mode(embedding_mat) <- "double"
  
  embedding_mat
}

embedding_rows <- split(
  seq_len(nrow(sample_text_tbl)),
  ceiling(seq_len(nrow(sample_text_tbl)) )
)

embedding_list <- map(
  embedding_rows,
  ~ get_embedding_batch(sample_text_tbl$review_text[.x])
)

embedding_mat <- do.call(rbind, embedding_list)

colnames(embedding_mat) <- paste0("emb_", seq_len(ncol(embedding_mat)))

embedding_tbl <- as_tibble(embedding_mat) %>%
  mutate(doc_id = sample_text_tbl$doc_id) %>%
  select(doc_id, everything())

# save embeddings because they take a long time to create
# write_rds(embedding_tbl, "../out/embedding_tbl.rds")

# Tokenization and Topic Modeling
corpus <- VCorpus(VectorSource(sample_text_tbl$review_text))

corpus_clean <- corpus %>%
  tm_map(content_transformer(str_to_lower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(content_transformer(lemmatize_strings)) %>%
  tm_map(
    removeWords,
    c(
      stopwords("en"),
      "company", "companies", "glassdoor",
      "job", "jobs", "work", "employee", "employees"
    )
  ) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus_clean)

# check the N/k ratio
n_docs <- nrow(dtm)
n_terms <- ncol(dtm)
n_docs / n_terms
# 0.9022651

sparsitycutoff <- 0.9997
slim_dtm <- removeSparseTerms(dtm, sparsitycutoff)
row_totals <- slam::row_sums(slim_dtm)

slim_doc_ids <- as.integer(slim_dtm$dimnames$Docs)

n_docs_slim <- nrow(slim_dtm)
n_terms_slim <- ncol(slim_dtm)
n_docs_slim / n_terms_slim
# 2.471083

token_mat <- as.matrix(slim_dtm)
colnames(token_mat) <- make.names(colnames(token_mat), unique = TRUE) # make token column names valid R names; names are unique but not all syntactically valid
token_tbl <- as_tibble(token_mat) %>% # convert token matrix to tibble 
  mutate(doc_id = slim_doc_ids) %>% # attach document IDs in the exact row order
  select(doc_id, everything()) # keep doc_id first for readability

dtm_stm <- readCorpus(slim_dtm, type = "slam")
detectCores()
# 8
cluster <- makePSOCKcluster(num_cores)
registerDoParallel(cluster)
kresult <- searchK(
  dtm_stm$documents, 
  dtm_stm$vocab, 
  K = seq(2, 20, by = 2), 
  cores = min(4, max(1, 8 - 1)) # use up to 4 cores instead of all available cores so the laptop stays stable becuase I first chose to use 7 cores. MY LAPTOP DIED
)



