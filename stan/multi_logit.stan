data {
  int<lower=0> N; // length
  int<lower=0> P; // parameters
  real<lower=0> alcohol[N]; //
  real <lower=0> acidity[N];
  array[N] int quality;
  matrix[N, P] X;
}

parameters {
  matrix[N, P] beta;
}

model {
  matrix[N, K] 
}
