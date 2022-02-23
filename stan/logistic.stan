data{ 
  int<lower = 1> N_train;
  vector[N_train] x1_train;
  vector[N_train] x2_train;
  int<lower = 0, upper = 1> y_train[N_train];
  int<lower = 1> N_test;
  vector[N_test] x1_test;
  vector[N_test] x2_test;
}

parameters {
  real beta0;
  real beta1;
  real beta2;
}

model {
  y_train ~ bernoulli_logit(beta0 + beta1 * x1_train + beta2 * x1_train);
  beta0 ~ normal(1, 0.25);
  beta1 ~ normal(5, 1);
  beta2 ~ normal(2, 1);
}

generated quantities {
  vector[N_test] y_test;
  for(i in 1:N_test) {
    y_test[i] = bernoulli_rng(inv_logit(beta0 + beta1 * x1_test[i] + beta2 * x2_test[i]));
  }
}
