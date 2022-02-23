library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)
# Create some fake data - logistic regression
set.seed(56)
N <- 500
beta0 <- 0.120
beta1 <- -0.75
beta2 <- 1.89

x1 <- runif(N, 0, 2)
x2 <- runif(N, 4, 8)

prob <- 1/(1 + exp(-(beta0 + beta1 * x1 + beta1 * x2)))

data.frame(x1, x2, y)
y <- rbinom(N, 1, prob)
# Distribution of y
table(y)

N_train <- N*0.8
N_test <- N*0.2
train_ind <- sample(c(1:N), size = N_train, replace = FALSE)

x1_train <- x1[train_ind]
x2_train <- x2[train_ind]
x1_test <- x1[-train_ind]
x2_test <- x2[-train_ind]
y_train <- y[train_ind]
y_test <- y[-train_ind]

data_list <- list(x1_train = x1_train,
                  x2_train = x2_train,
                  y_train = y_train, 
                  N_train = N_train,
                  x1_test = x1_test, 
                  x2_test = x2_test, 
                  N_test = N_test)
plot(x2, y)


data_file_path <- "data/stan_data/logistic_sim_data.R"
stan_rdump(names(data_list), file = data_file_path)
input_data <- read_rdump(data_file_path)

fit <- stan(file = "stan/logistic.stan",
            data = input_data,
            chains = 10, iter = 5000)

print(fit)
plot(fit, pars = paste0("beta", 0:2))
traceplot(fit, pars = paste0("beta", 0:2))

print(fit, pars = paste0("beta", 0:2))

ext_fit <- extract(fit)
print(fit, pars = "beta1")
median(ext_fit$beta0) - beta0
median(ext_fit$beta1) - beta1
median(ext_fit$beta2) - beta2
hist(ext_fit$beta1)
glm(y ~ x1 + x2, family = binomial(link = logit))

sum(apply(ext_fit$y_test, 2, median) == y_test) / length(y_test)
# https://medium.com/@alex.pavlakis/making-predictions-from-stan-models-in-r-3e349dfac1ed
