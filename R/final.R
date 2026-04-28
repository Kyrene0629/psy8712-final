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
set.seed(1234)


