# Script Settings and Resources
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))  # this sets the working directory to the final folder that has the current R script and all the other stuff. it is convenient.
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
set.seed(1234) # I set the seed to 1234 for reproducibility

# Data Import and Cleaning
glassdoor_raw_tbl <- read_csv("../data/glassdoor_reviews.csv") # This imports the Glassdoor review dataset from the data subfolder. I use read_csv() instead of base R read.csv() because read_csv() is faster, returns a tibble, and gives a cleaner column parsing messages. I did not use fread() becuase, later on, I used tydiverse syntax.

ml_text_tbl <- glassdoor_raw_tbl %>% # begins a tidyverse pipeline
  select(overall_rating, headline, pros, cons) %>% # keeps only the outcome variable and the available text fields that are needed, I chose select() because the full dataset has many columns that we dont need
  mutate(review_id = row_number(), # creates a unique review ID that is based on row order. This can be used to track reviews after sampling, joining...
         review_text = str_squish(str_c( # creates a single review text field. str_squish is used to remove whitespace, and str_c is used to concatenate multiple text fields
           replace_na(headline, ""), # The following three lines replace missing text with empty strings. This can prevengt NA from causing the entire combined review text to become NA. I did not choose to delete missing rows because a review can still have useful text in some of the three fields. (try  ot to be aggressive)
           replace_na(pros, ""),
           replace_na(cons, ""),
           sep = " " # inserts spaces between the headline, pros, and cons. This so the words dont incorrectly blend together
         ))
         ) 
# ml_text_tbl$review_text this is diagnostic check for myself
# nrow(ml_text_tbl) this is another diagnostic check (the imported dataset has 838,566 reviews)
# 838566

sample_n <- 10000 # fit ML models with 838566 reviews is way to slow, so I set the target sample size.
sample_prop <- sample_n / nrow(ml_text_tbl) # converts the desired sample size into a sampling proportion. Later, in the next pipeline slice_sample(prop = sample_prop) samples the same proportion within each rating group
# 0.01192512

sample_text_tbl <- ml_text_tbl %>% # start a new pipeline for creating the modeling sample 
  group_by(overall_rating) %>% # groups reviews by satisfaction rating, and it allows stratified sampling by the outcome variable
  slice_sample(prop = sample_prop) %>% # samples the same proportion of reviews from each rating level instead of simple random sampling because it keeps the original rating distribution more reliably
  ungroup() %>% # this removes raitng grouping because later operations should apply to the whole dataset not within each rating group
  mutate(doc_id = row_number()) %>% # creates the document ID for the sampled text
  select(doc_id, overall_rating, review_text) # only keep the varibales that  are needed for modeling. I put doc_id first so joins will be eaiser to inspect

# check the sample size and rating distribution
# nrow(sample_text_tbl)
# 9998
# the reason it is not 10000 is becuase proportional stratified sampling with integer-sized rating groups can round smaller.

sample_text_tbl %>% # this checks the rating distribution in the sample. I use count() to calculate frequencies and mutate() to calculate proportions, so I can confirm that the sample still represents the outcome distribution
  count(overall_rating) %>%
  mutate(prop = n / sum(n))

# Embeddings
get_embedding_batch <- function(text_vec) { # sends a vector of texts to Ollama and return the embeddings, I used this becuase I need to repeat the same API request many times across batches.
  response <- POST( # send a HTTP POST request, so the text input is sent in the request not in the URL
    url = "http://localhost:11434/api/embed",
    content_type_json(), # set the content type and tell tell Ollama the request body is JSON, not plain text 
    body = list( # defines the JSON body sent to Ollama by using nomic-embed-text and input is the review text
      model = "nomic-embed-text",
      input = text_vec
    ),
    encode = "json" # encode the R list to JSON because API cannot directly interpret R list objects
  )
  
  stop_for_status(response) # the script would stop if the API request fails, make sure the request wouod not corrupt the embedding
  
  result <- content(response, as = "parsed") # parse the API response into an R object
  embedding_mat <- do.call(rbind, result$embeddings) # rowbinds the embeddings into a numeric matrix. each row is one review and each column would be one embedidng dimension
  storage.mode(embedding_mat) <- "double" # make sure the matrix is stored as numeric double precision. machine leanring models require nuneric predictors
  
  embedding_mat # returns the embedding matrix
}

embedding_batch_size <- 50 # set the number of reviews sent to Ollama per request, I choose 50 becuase it balance the runtime and Ollama stability

embedding_rows <- split( # these few lines create groups of row indices for batching. seq_len() creates the row numbers, ceiling() then assigns each row to a batch, and finally split() divides the row numbers into the batch lists
  seq_len(nrow(sample_text_tbl)),
  ceiling(seq_len(nrow(sample_text_tbl)) / embedding_batch_size)
)

embedding_list <- map( # then the embedding function is applied to each batch of review text. map() is used becuase each batch returns a matrix and those results are stored in a list before combining
  embedding_rows,
  ~ get_embedding_batch(sample_text_tbl$review_text[.x])
)

embedding_mat <- do.call(rbind, embedding_list) # this combines all batch-level embedding matrices into one embedding matrix. Row order is preserved because embedding_rows are created in order

colnames(embedding_mat) <- paste0("emb_", seq_len(ncol(embedding_mat)))  # The embedding dimensions would be emb_1, emb_2... ML models and tibbles need interpretable column names. 

embedding_tbl <- as_tibble(embedding_mat) %>% # convert the embedding matrix into a tibble so that later joins and modeling datasets will be easier to manage in tidyverse forms.
  mutate(doc_id = sample_text_tbl$doc_id) %>% # matches document IDs to the embeddings, so later embeddings can be joined back to the outcome and other 
  select(doc_id, everything()) # place doc_id first so it is easier to read

# save embeddings because they take a long time to create
# write_rds(embedding_tbl, "../out/embedding_tbl.rds")

# Tokenization and Topic Modeling
corpus <- VCorpus(VectorSource(sample_text_tbl$review_text)) # create a corpus from the sampled review text for tm text cleaning

corpus_clean <- corpus %>% # this is the text cleaning pipeline
  tm_map(content_transformer(str_to_lower)) %>% # make everyhting lowercases so that M/m can be treated as the same token
  tm_map(removePunctuation) %>% # remove puctuation becuase they dont represent any important content
  tm_map(removeNumbers) %>% # remove numbers just to reduce noises. they dont generalize well as text
  tm_map(content_transformer(lemmatize_strings)) %>% # lemmatization produces more reabable word forms
  tm_map( # the following is to remove common stopwords and terms that are too generic in the employee reviews to distinguish satisfaction
    removeWords,
    c(
      stopwords("en"),
      "company", "companies", "glassdoor",
      "job", "jobs", "work", "employee", "employees"
    )
  ) %>%
  tm_map(stripWhitespace) # remove extra whitespace after cleaning so no empty spaces

dtm <- DocumentTermMatrix(corpus_clean) # create a document term matrix that rows are reviews, coumns are tokens, cell are token counts. This document term matrix can work with tm, stm, and later machine leanring construction

# check the N/k ratio
n_docs <- nrow(dtm)
n_terms <- ncol(dtm)
n_docs / n_terms
# 0.9022651

sparsitycutoff <- 0.9997 # I set a very lenient threshold. this removes terms that are absent from more than 99.97% of documents
slim_dtm <- removeSparseTerms(dtm, sparsitycutoff) # this removes extremely sparse terms to remove noises and highly dimensional predictors. make sure they dont slwo down topic modeling and later the machine learning part.


slim_doc_ids <- as.integer(slim_dtm$dimnames$Docs) # this extracts document IDs from the slimed DTM, so when dtm processes, I can know which original review coresspond to which.

# check the n/k ratio after sparsity filtering
n_docs_slim <- nrow(slim_dtm)
n_terms_slim <- ncol(slim_dtm)
n_docs_slim / n_terms_slim
# 2.471083, becomes more manageable

token_mat <- as.matrix(slim_dtm) # convert the DTM into a matrix for later joining into a tibble and for the mahcine leanring part
colnames(token_mat) <- make.names(colnames(token_mat), unique = TRUE) # make token column names valid R names; names are unique but not all syntactically valid
token_tbl <- as_tibble(token_mat) %>% # convert token matrix to tibble 
  mutate(doc_id = slim_doc_ids) %>% # attach document IDs in the exact row order
  select(doc_id, everything()) # keep doc_id first for readability

dtm_stm <- readCorpus(slim_dtm, type = "slam") # convert into the list format required by stm
detectCores() # check how many cpu cores in my computer
# 8
num_cores <- 8 - 1 # set the number of cores to 7, use most but not all cpu cores
registerDoParallel(num_cores) # this is a parallel backend so multiple cpu cores can be used

kresult <- searchK( # topic models with topics from 2 to 20, searchK() is used so the number selection is based on diagnostics
  dtm_stm$documents, 
  dtm_stm$vocab, 
  K = seq(2, 20, by = 2) 
)
stopImplicitCluster() # stop the parallel backend to release the cpu cores

plot(kresult) # use it to inspect the topic number diagnostics, this is the visual display
kresult$results # this gives the numerical diagnostics

# select K = 10 becuas it has the best heldout likelihood and has strong exclusivity and a manageable number of interpretable topics

topic_model <- stm( # so then I fit the final STM topic model with 10 topics, these topic proportions can then be used as predictors of overall satisfaction
  documents = dtm_stm$documents,
  vocab = dtm_stm$vocab,
  K = 10
)

topic_labels <- labelTopics(topic_model, n = 10) # this extracts the top words for each topic so I can know what each topic represents

topic_examples <- findThoughts( # this finds representative review texts for each topic and match() is used to align the sampled text with the documents in slim_dtm
  topic_model,
  texts = sample_text_tbl$review_text[match(slim_doc_ids, sample_text_tbl$doc_id)],
  n = 3
)

topic_corr <- topicCorr(topic_model) # compute correlation among topics

theta <- topic_model$theta # extract the estimated proportion of that review devoted to the topic.

# for RQ 2 & 3
topic_tbl <- as_tibble(theta, # this converts the topic proportion matrix into a tibble and name each column has each name like topic1, topic2, and move the document IDs. I used .name_repair argument to avoid the warning from unnamed matrix columns
                       .name_repair = ~ paste0("topic_", seq_along(.x))) %>% 
  mutate(doc_id = slim_doc_ids) %>% 
  select(doc_id, everything())

# Final ML Dataset

ml_tbl <- sample_text_tbl %>%
  inner_join(token_tbl, by = "doc_id") %>%
  inner_join(embedding_tbl, by = "doc_id") %>%
  inner_join(topic_tbl, by = "doc_id") %>%
  select(-review_text)
# saveRDS(ml_tbl, "../out/data.RDS")

ml_id_tbl <- ml_tbl %>%
  select(doc_id, overall_rating)

token_df <- ml_id_tbl %>%
  inner_join(token_tbl, by = "doc_id")

embedding_df <- ml_id_tbl %>%
  inner_join(embedding_tbl, by = "doc_id")

topic_df <- ml_id_tbl %>%
  inner_join(topic_tbl, by = "doc_id")

token_topic_df <- ml_id_tbl %>%
  inner_join(token_tbl, by = "doc_id") %>%
  inner_join(topic_tbl, by = "doc_id")

embedding_topic_df <- ml_id_tbl %>%
  inner_join(embedding_tbl, by = "doc_id") %>%
  inner_join(topic_tbl, by = "doc_id")

all_feature_df <- ml_id_tbl %>%
  inner_join(token_tbl, by = "doc_id") %>%
  inner_join(embedding_tbl, by = "doc_id") %>%
  inner_join(topic_tbl, by = "doc_id")

# Analysis
train_index <- createDataPartition(
  ml_tbl$overall_rating,
  p = .8,
  list = FALSE
)

train_tbl <- ml_tbl[train_index, ]
test_tbl <- ml_tbl[-train_index, ]

training_token <- token_df[train_index, ] %>%
  select(-doc_id) 

holdout_token <- token_df[-train_index, ] %>%
  select(-doc_id) 

training_embedding <- embedding_df[train_index, ] %>%
  select(-doc_id) 

holdout_embedding <- embedding_df[-train_index, ] %>%
  select(-doc_id) 

training_topic <- topic_df[train_index, ] %>%
  select(-doc_id) 
holdout_topic <- topic_df[-train_index, ] %>%
  select(-doc_id) 

training_token_topic <- token_topic_df[train_index, ] %>%
  select(-doc_id) 

holdout_token_topic <- token_topic_df[-train_index, ] %>%
  select(-doc_id) 

training_embedding_topic <- embedding_topic_df[train_index, ] %>%
  select(-doc_id) 

holdout_embedding_topic <- embedding_topic_df[-train_index, ] %>%
  select(-doc_id) 

training_all <- train_tbl %>%
  select(-doc_id) 

holdout_all <- test_tbl %>%
  select(-doc_id) 

# Elastic Net Models
fit_token <- train(
  overall_rating ~ .,
  data = training_token, 
  method = "glmnet",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv", "center", "scale"), 
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)),  
  metric = "RMSE" 
)

fit_embedding <- train(
  overall_rating ~ ., 
  data = training_embedding, 
  method = "glmnet",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv", "center", "scale"), 
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)), 
  metric = "RMSE" 
)

fit_topic <- train(
  overall_rating ~ ., 
  data = training_topic, 
  method = "glmnet", 
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv", "center", "scale"), 
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)), 
  metric = "RMSE" 
)

fit_token_topic <- train(
  overall_rating ~ ., 
  data = training_token_topic, 
  method = "glmnet", 
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv", "center", "scale"), 
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)), 
  metric = "RMSE" 
)

fit_embedding_topic <- train(
  overall_rating ~ ., 
  data = training_embedding_topic, 
  method = "glmnet", 
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv", "center", "scale"), 
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)),
  metric = "RMSE" 
)

fit_all <- train(
  overall_rating ~ ., 
  data = training_all, 
  method = "glmnet", 
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv", "center", "scale"), 
  tuneGrid = expand.grid(
    alpha = c(0, 1),
    lambda = seq(0.0001, 0.1, length = 10)), 
  metric = "RMSE" 
)

# Random Forest Model
fit_embedding_topic_rf <- train(
  overall_rating ~ ., 
  data = training_embedding_topic, 
  method = "ranger", 
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE 
  ), 
  preProcess = c("zv"), 
  tuneGrid = expand.grid(
    mtry = c(25, 100, 250), 
    splitrule = "variance", 
    min.node.size = c(5, 10)), 
  metric = "RMSE"
)

