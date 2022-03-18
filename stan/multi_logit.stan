data {
   int N_train; 
   int N_test; 
   int K;                       // number of classees
   int D;                       // number of predictors
   int y_train[N_train];
   row_vector[D] X_train[N_train];
   row_vector[D] X_test[N_test];
}

parameters {
   vector[D] beta;
   ordered[K-1] c;
}

model {
   // prior
 beta ~ normal(0, 1); 
 
 for (i in 1:N_train)
   y_train[i] ~ ordered_logistic(X_train[i] * beta, c);
}

generated quantities {
  int<lower=1, upper=K> y_test[N_test];

  for (i in 1:N_test)
    y_test[i] = ordered_logistic_rng(X_test[i] * beta, c);

}
