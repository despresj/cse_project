library(rstan) # observe startup messages
options(mc.cores = parallel::detectCores())

rstan_options(auto_write = TRUE)

n <- 550
beta0 <- .2
beta1 <- .12
beta2 <- 0.25
sigma <- 1

epsilon <- rnorm(n, 0, sigma) + rexp(n, 50)

x1 <- runif(n, 40, 90)
x2 <- runif(n, 10, 100)

y <- beta0 +  beta1 * x1 + beta2 * x2 + epsilon
plot(x, y)

gen_dat <- list(
  J = n,
  y = y,
  x1 = x1,
  x2 = x2
)

fit <- stan(file = "stan/schools.stan",
            data = gen_dat, 
            iter = 1000, chains = 4)
print(fit)

plot(fit, pars = paste0("beta", 0:2))

traceplot(fit, pars = paste0("beta", 0))
predict(fit)

ext_fit <- extract(fit)  