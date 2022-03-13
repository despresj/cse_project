library(rstan) # observe startup messages
options(mc.cores = parallel::detectCores())

df <- read.csv("data/clean/wine_quality.csv")
train <- sample(1:nrow(df), size = floor(nrow(df) * 0.8), replace = FALSE)
df_train <- df[train,]
df_test <- df[-train,]

covariates <- names(df)[names(df) != "quality"]
formula <- as.formula(paste("quality ~ ", paste0(covariates, collapse = " + ")))
X_train <- model.matrix(formula, df_train)
X_test <- model.matrix(formula, df_test)

wine_data <- list(
  N_train = nrow(X_train),
  N_test = nrow(X_test),
  K = length(unique(df_train$quality)),
  D = ncol(X),
  y_train = as.numeric(factor(df_train$quality)),
  y_test = as.numeric(factor(df_test$quality)),
  X_train = X_train,
  X_test = X_test
)

fit <- stan("stan/multi_logit.stan",
            data=wine_data,
            chains = 8, iter = 250, warmup = 50)
print(fit)
traceplot(fit)
