library("rstan") # observe startup messages
options(mc.cores = parallel::detectCores())

df <- readr::read_csv("data/clean/wine_quality.csv")



# wine_data <- list(
#   N = nrow(df),
#   P = 3,
#   quality = as.integer(df$quality),
#   alcohol = df$alcohol,
#   acidity = df$fixed_acidity
# )
# 
# fit <- stan("stan/multi_logit.stan", 
#             data=wine_data, 
#             chains = 1, iter = 3000, warmup = 500, thin = 10)
