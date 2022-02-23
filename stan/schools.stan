// saved as schools.stan
data {
  int<lower=0> J;         
  vector[J] x1;
  vector[J] x2;
  vector[J] y;
}

parameters {
  real beta0;
  real beta1;
  real beta2;
  real<lower=0> sigma;
}

model {
  y ~ normal(beta0 + beta1 * x1 + beta2 * x2, sigma);
}
