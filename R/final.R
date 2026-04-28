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
         review_text = str_squish(paste(replace_na(headline, ""),
                                        replace_na(pros, ""),
                                        replace_na(cons, ""),
                                   sep = " ")
                                  )
         )
ml_text_tbl$review_text

