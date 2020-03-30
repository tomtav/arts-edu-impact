# load packages in this order
library(lsmeans)
library(emmeans)
library (lmerTest)
library (effects)
library(car)
library(ggpubr)
library(Rmisc)

# 
getwd()
setwd("/Users/katelinmaguire/Desktop")

# cleaned data
ela_funding <- read.csv("ela_funding.csv")
math_funding <- read.csv("math_funding.csv")

# pass rate data
ela_pass_rate_df <- read.csv("ela_pass_rate_df.csv")
math_pass_rate_df <- read.csv("math_pass_rate_df.csv")

# look at dfs
str(ela_funding)
str(math_funding)

# look at dfs
str(ela_pass_rate_df)
str(math_pass_rate_df)

# change type to factor
to_factors <- c("Year", "Q1_funding_sources", "year")
ela_pass_rate_df[, to_factors] <- lapply(ela_pass_rate_df[, to_factors], factor)

to_factors <- c("Year", "Q1_funding_sources", "year")
math_pass_rate_df[, to_factors] <- lapply(math_pass_rate_df[, to_factors], factor)
