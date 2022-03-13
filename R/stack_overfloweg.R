library(rstan)
n <- 100
set.seed(1)
dat <- data.frame(DV=factor(sample(letters[1:5], n, replace=T)),
                  x1 = rnorm(n, mean=4.5, sd=1.3),
                  x2 = sample(c(1:5), prob=c(0.035, 0.167, 0.083, 0.415, 0.298)),
                  x3 = sample(c(0,1), prob=c(.51, .49)),
                  x4 = round(rnorm(n, mean=48, sd=15),0))
stan_model <- "
 data {
 int K;
 int N; 
 int D;
 int y[N];
 matrix[N, D] x;
 }
 parameters {
 matrix[D, K] beta;
 }
 model {
 matrix[N, K] x_beta = x * beta;

 to_vector(beta) ~ normal(0, 2);

 for (n in 1:N)
 y[n] ~ categorical_logit(x_beta[n]');
 }

"

rstan_options(auto_write = TRUE)
options(mc.cores = 4)
f <- as.formula("DV ~ x1 + x2 + x3 + x4")
M <- model.matrix(f, dat)
#data for stan
datlist <- list(N=nrow(M),                     #nr of obs
                K=length(unique(dat[,1])),     #possible outcomes
                D=ncol(M),                     #dimension of predictor matrix
                x=M,                           #predictor matrix
                y=as.numeric(dat[,1]))

#estimate model
b.out <- stan(model_code=stan_model, 
              data=datlist,
              iter = 1000,
              chains = 4,
              seed = 12591,
              control = list(max_treedepth = 11))
