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
# the reason it is not 10000 is because proportional stratified sampling with integer-sized rating groups can round smaller.

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
registerDoSEQ() # force caret to use sequential processing to reduce memory copying

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

topic_tbl <- as_tibble(theta, # this converts the topic proportion matrix into a tibble and name each column has each name like topic1, topic2, and move the document IDs. I used .name_repair argument to avoid the warning from unnamed matrix columns
                       .name_repair = ~ paste0("topic_", seq_along(.x))) %>% 
  mutate(doc_id = slim_doc_ids) %>% 
  select(doc_id, everything())

# Analysis

# Final ML Dataset

ml_tbl <- sample_text_tbl %>% # to create the final machine leaning table, I first start from the sample review dataset
  inner_join(token_tbl, by = "doc_id") %>% # then join token predictors by doc_id, I use inner_join() becuase it only keeps only documents with valid token rows
  inner_join(embedding_tbl, by = "doc_id") %>% # then join embedding predictors by doc_id, I still use inner_join() because it keeps the machine learning dataset that aligned across feature sets
  inner_join(topic_tbl, by = "doc_id") %>% # join topic predictors by doc_id, so this creates the full pre-split machine leanring dataset that is necessary for the following parts 
  select(-review_text) # remove review text because machine learning models only use numeric predictors not text
# saveRDS(ml_tbl, "../out/data.RDS") # save the machine learning dataset

ml_id_tbl <- ml_tbl %>%  # make a small table with only ID and outcome, so later joins can be cleaner
  select(doc_id, overall_rating) # keep those two as the outcomes

token_df <- ml_id_tbl %>% # token only dataset, this is the baseline for RQ1 and RQ2
  inner_join(token_tbl, by = "doc_id") # use inner_join by doc_id to protect row matching

embedding_df <- ml_id_tbl %>% # embedding only the dataset
  inner_join(embedding_tbl, by = "doc_id") # join embeddings back to the same review ids

token_embedding_df <- ml_id_tbl %>% # token + embedding dataset, needed for RQ1
  inner_join(token_tbl, by = "doc_id") %>% # add token predictors first because tokenization is the baseline
  inner_join(embedding_tbl, by = "doc_id") # then add embeddings to see if they can improve beyond tokens

topic_df <- ml_id_tbl %>% # topic only dataset, tests only topics
  inner_join(topic_tbl, by = "doc_id") # join topic probability back to the same review ids

token_topic_df <- ml_id_tbl %>% # token + topic dataset, needed for RQ2
  inner_join(token_tbl, by = "doc_id") %>% # add token predictors as the baseline
  inner_join(topic_tbl, by = "doc_id") # then add topic predictors to see if topics can improve beyond tokens

embedding_topic_df <- ml_id_tbl %>% # embedding + topic dataset, needed for RQ3 and nonlinear models
  inner_join(embedding_tbl, by = "doc_id") %>% # add embeddings because they represent semantic meaning
  inner_join(topic_tbl, by = "doc_id") # then add topics because they represent main ideas in the reviews

all_feature_df <- ml_id_tbl %>% # used for RQ4
  inner_join(token_tbl, by = "doc_id") %>% # include tokens
  inner_join(embedding_tbl, by = "doc_id") %>% # include embeddings
  inner_join(topic_tbl, by = "doc_id") # include topics

train_index <- createDataPartition( # create train index. I use createDataPartition() to keep the rating distribution more balanced
  ml_tbl$overall_rating, # split based on outcome variable
  p = .8, # use 80% for training and 20% for holdout
  list = FALSE # return row numbers instead of a list, so it is easier to subset later
)

train_tbl <- ml_tbl[train_index, ] # create training data from the 80% selected rows
test_tbl <- ml_tbl[-train_index, ] # create holdout data from the other 20% of rows

training_token <- token_df[train_index, ] %>% # create training data for the token model
  select(-doc_id)  # remove doc_id because it is not a predictor

holdout_token <- token_df[-train_index, ] %>% # create holdout data for the token model
  select(-doc_id) # remove doc_id to match with the training data

training_embedding <- embedding_df[train_index, ] %>% # create training data for the embedding model
  select(-doc_id) # remove doc_id because it doesnt predict ratings

holdout_embedding <- embedding_df[-train_index, ] %>% # create holdout data for the embedding model
  select(-doc_id) # remove doc_id,so it only use the embedding predictors

training_token_embedding <- token_embedding_df[train_index, ] %>%  # create training data for token + embedding model
  select(-doc_id) # remove doc_id because it doesnt have meaningful text information

holdout_token_embedding <- token_embedding_df[-train_index, ] %>% # create holdout data for token + embedding model
  select(-doc_id) # remove doc_id to match with the training data

training_topic <- topic_df[train_index, ] %>% # create training data for topic model
  select(-doc_id) # keep only predictors
holdout_topic <- topic_df[-train_index, ] %>% # create holdout data for topic model
  select(-doc_id) # keep only predictors

training_token_topic <- token_topic_df[train_index, ] %>% # create training data for token + topic model
  select(-doc_id) # keep only predictors

holdout_token_topic <- token_topic_df[-train_index, ] %>% # create holdout data for token + topic model
  select(-doc_id) # keep only predictors

training_embedding_topic <- embedding_topic_df[train_index, ] %>% # create training data for embedding + topic model
  select(-doc_id) # keep only predictors

holdout_embedding_topic <- embedding_topic_df[-train_index, ] %>% # create holdout data for embedding + topic model
  select(-doc_id) # keep only predictors

training_all <- train_tbl %>% # create training data with all predictors
  select(-doc_id) # keep only predictors

holdout_all <- test_tbl %>% # create holdout data with all predictors
  select(-doc_id) # keep only predictors

# Elastic Net Models
fit_token <- train(
  overall_rating ~ ., # predict overall rating from all token predictors
  data = training_token,  # use token training data 
  method = "glmnet", # use glmnet because there are many predictors and they are correlated, ols is not suitable here
  trControl = trainControl(
    method = "cv",# use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv", "center", "scale"), # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # alpha 0 is ridge and alpha 1 is lasso
    lambda = seq(0.0001, 0.1, length = 10)),  # examine different penalty values
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

fit_embedding <- train(
  overall_rating ~ ., # predict overall rating from the embedding dimensions
  data = training_embedding, # use embedding training data to test the embeddings 
  method = "glmnet",  # use glmnet because the embeddings are numeric predictors and can be correlated
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv", "center", "scale"), # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # compare ridge and lasso
    lambda = seq(0.0001, 0.1, length = 10)), # examine different penalty values
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

fit_token_embedding <- train(
  overall_rating ~ ., # predict overall rating from both tokens and embeddings
  data = training_token_embedding, # it tests the embeddings beyond tokens, sue for RQ1
  method = "glmnet", # use glmnet because the embeddings are numeric predictors and can be correlated
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ),
  preProcess = c("nzv", "center", "scale"),  # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # compare ridge and lasso
    lambda = seq(0.0001, 0.1, length = 10) # examine different penalty values
  ),
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

fit_topic <- train(
  overall_rating ~ ., # predict overall rating from topic probabilities
  data = training_topic,  # use topic data to test topics alone
  method = "glmnet", # use glmnet because the embeddings are numeric predictors and can be correlated
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5,# use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv", "center", "scale"), # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # compare ridge and lasso
    lambda = seq(0.0001, 0.1, length = 10)), # examine different penalty values
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

fit_token_topic <- train(
  overall_rating ~ ., # predict overall rating from tokens + topics
  data = training_token_topic, # it tests topics beyond tokens, use for RQ2
  method = "glmnet", # tokens are highly dimensional
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv", "center", "scale"), # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # compare ridge and lasso
    lambda = seq(0.0001, 0.1, length = 10)), # examine different penalty values
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

fit_embedding_topic <- train(
  overall_rating ~ ., # predict overall rating from embeddings + topics
  data = training_embedding_topic, # it combines two reduced text features, use for RQ3
  method = "glmnet", # embeddings and topics may still be correlated
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv", "center", "scale"), # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # compare ridge and lasso
    lambda = seq(0.0001, 0.1, length = 10)), # examine different penalty values
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

fit_all <- train(
  overall_rating ~ ., # predict overall rating from tokens, embeddings, and topics
  data = training_all, # use all for RQ4
  method = "glmnet", # highly dimensional
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv", "center", "scale"), # remove near zero variance predictors and standardize predictors for glmnet
  tuneGrid = expand.grid(
    alpha = c(0, 1), # compare ridge and lasso
    lambda = seq(0.0001, 0.1, length = 10)),  # examine different penalty values
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

# Random Forest Model
fit_embedding_topic_rf <- train(
  overall_rating ~ ., # predict overall rating from embeddings + topics
  data = training_embedding_topic, 
  method = "ranger", # use ranger because it is the random forest method in caret
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE # print progress
  ), 
  preProcess = c("nzv"), # remove near zero variance predictors, and no center and scale because trees do not need that
  tuneGrid = expand.grid(
    mtry = c(25, 100, 250), # test different numbers of predictors available at each split
    splitrule = "variance", # because overall_rating is numeric
    min.node.size = c(5, 10)), # test two node sizes, it affects tree complexity
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
)

# XGBoost Tree Model
fit_embedding_topic_xgbtree <- train(
  overall_rating ~ ., # predict overall rating from embeddings + topics
  data = training_embedding_topic, # keep xgboost runtime manageable
  method = "xgbTree", # it is a strong boosted tree model in caret
  trControl = trainControl(
    method = "cv", # use cross validation
    number = 5, # use 5 folds, 10 folds have longer runtime
    verboseIter = TRUE), # print progress
  preProcess = c("nzv"), # remove near zero variance predictors, and no center and scale because trees do not need that
  tuneGrid = expand.grid(nrounds = c(50, 100), # test number of boosting rounds
                         max_depth = c(2, 4), # avoid too much overfitting
                         eta = c(0.05, 0.1), # alswo learning rates
                         gamma = 0, # keep gamma fixed to keep grid smaller
                         colsample_bytree = 0.8, # use 80% of predictors for each tree
                         min_child_weight = 1, # use default to avoid making grid too large
                         subsample = 0.8), # use 80% of rows for each tree
  metric = "RMSE" # overall_rating is numeric, so I use RMSE
  )


# I only examined embeddings + topics for random forest and XGBoost Tree Model for extra comparisons for RQ4. I did not try every set of comparisons becuase these models can be much slower with the whole token matrix.
# I think I made a very practical choice.
# I included all elastic net models becuase elastic net handles high dimensional predictors very well.

# Model Comparison
model_resamples <- resamples(
  list(
    token_glmnet = fit_token, # token only model
    embedding_glmnet = fit_embedding, # embedding only model
    topic_glmnet = fit_topic, # topic only model
    token_embedding_glmnet = fit_token_embedding, # token + embedding model for RQ1
    token_topic_glmnet = fit_token_topic, # token + topic model for RQ2
    embedding_topic_glmnet = fit_embedding_topic, # embedding + topic model for RQ3
    all_glmnet = fit_all, # elastic net model for RQ4
    embedding_topic_rf = fit_embedding_topic_rf, # random forest model
    embedding_topic_xgbtree = fit_embedding_topic_xgbtree # xgboost model
  )
)
summary(model_resamples) # summarize al models' cross validated performance 
dotplot(model_resamples, metric = "RMSE") # plot RMSE from resamples

# holdout evaluation
token_pred <- predict(fit_token, holdout_token) # predict holdout ratings from the token model
embedding_pred <- predict(fit_embedding, holdout_embedding)  # predict holdout ratings from the embedding model
topic_pred <- predict(fit_topic, holdout_topic) # predict holdout ratings from the topic model
token_embedding_pred <- predict(fit_token_embedding, holdout_token_embedding) # predict holdout ratings from token + embedding model
token_topic_pred <- predict(fit_token_topic, holdout_token_topic) # predict holdout ratings from token + topic model
embedding_topic_pred <- predict(fit_embedding_topic, holdout_embedding_topic) # predict holdout ratings from embedding + topic model
all_pred <- predict(fit_all, holdout_all) # predict holdout ratings from all feature model
rf_pred <- predict(fit_embedding_topic_rf, holdout_embedding_topic) # predict holdout ratings from random forest
xgbtree_pred <- predict(fit_embedding_topic_xgbtree, holdout_embedding_topic) # predict holdout ratings from xgboost tree

holdout_results_tbl <- tibble( # I only want to create one table to store holdout performance for every model
  model_name = c( # store model names, ensure readability
    "Token elastic net",
    "Embedding elastic net",
    "Topic elastic net",
    "Token + embedding elastic net",
    "Token + topic elastic net",
    "Embedding + topic elastic net",
    "Token + embedding + topic elastic net",
    "Embedding + topic random forest",
    "Embedding + topic xgboost tree"
  ),
  base_model = c( # store the model type used for each row
    "glmnet",
    "glmnet",
    "glmnet",
    "glmnet",
    "glmnet",
    "glmnet",
    "glmnet",
    "ranger",
    "xgbTree"
  ),
  holdout_rmse = c( # calculate RMSE for each model, lower is better
    RMSE(token_pred, holdout_token$overall_rating),
    RMSE(embedding_pred, holdout_embedding$overall_rating),
    RMSE(topic_pred, holdout_topic$overall_rating),
    RMSE(token_embedding_pred, holdout_token_embedding$overall_rating),
    RMSE(token_topic_pred, holdout_token_topic$overall_rating),
    RMSE(embedding_topic_pred, holdout_embedding_topic$overall_rating),
    RMSE(all_pred, holdout_all$overall_rating),
    RMSE(rf_pred, holdout_embedding_topic$overall_rating),
    RMSE(xgbtree_pred, holdout_embedding_topic$overall_rating)
  ),
  holdout_rsq = c( # calculate R squared for each model, higher is better
    R2(token_pred, holdout_token$overall_rating),
    R2(embedding_pred, holdout_embedding$overall_rating),
    R2(topic_pred, holdout_topic$overall_rating),
    R2(token_embedding_pred, holdout_token_embedding$overall_rating),
    R2(token_topic_pred, holdout_token_topic$overall_rating),
    R2(embedding_topic_pred, holdout_embedding_topic$overall_rating),
    R2(all_pred, holdout_all$overall_rating),
    R2(rf_pred, holdout_embedding_topic$overall_rating),
    R2(xgbtree_pred, holdout_embedding_topic$overall_rating)
  )
) %>%
  arrange(holdout_rmse) # so best model appears first

holdout_results_tbl # print the holdout results
write_csv(holdout_results_tbl, "../out/holdout_results.csv") # save holdout results to the out subfolder

# Research Questions 1 ~ 4
# RQ1. Does the use of embeddings (using the nomic-embed-text LLM embeddings model) improve prediction of satisfaction beyond a rigorous tokenization strategy?
# Yes, the use of embeddings improve prediction of satisfaction beyond a rigorous tokenization strategy.
# The Token elastic net model has RMSE = 1.04 and R squared = 0.204.
# The Token + embedding elastic net model has RMSE = 0.888 and R squared = 0.414.
# Since RMSE decreases from 1.04 to 0.888, embeddings improves prediction beyond the rigorous tokenization model.

# RQ2. Does the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy?
# Yes, the use of topics improve prediction of satisfaction beyond a rigorous tokenization strategy, but the improvement is smaller than embeddings
# The Token elastic net model has RMSE = 1.04 and R squared = 0.204.
# The Token + topic elastic net model has RMSE = 0.996 and R squared = 0.263
# Since RMSE decreases from 1.04 to 0.996, topics improve prediction beyond tokenization. However, topics add some incremental value to the tokens, but topics themselves are weak predictors because the Topic elastic net model has RMSE = 1.04 and R squared = 0.191

# RQ3. Does the use of embeddings plus topics improve prediction of satisfaction beyond either alone?
# Yes, but the use of embeddings plus topics only slightly improve prediction of satisfaction beyond embeddings alone
# The Embedding elastic net model has RMSE = 0.899 and R squared = 0.400.
# The Topic elastic net model has RMSE = 1.04 and R squared = 0.191
# The Embedding + topic elastic net model has RMSE = 0.892 and R squared = 0.410
# Since 0.892 is lower than 0.899 and 1.04, the Embedding + topic elastic net model improves prediction beyond either one alone. However, I did notice the improvement beyond embeddings alone is pretty small

# RQ4. What is the best prediction of overall job satisfaction achievable using text reviews as source data?
# The best prediction of overall job satisfaction achievable using text reviews as source data is the Token + embedding + topic elastic net model becuase it has RMSE = 0.883 and R squared = 0.421
# The best prediction comes from combining tokens, embeddings, and topics in an elastic net model

