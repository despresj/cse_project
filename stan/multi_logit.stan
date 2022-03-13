data {
   int N_train; 
   int N_test; 
   int K;
   int D;
   int y_train[N_train];
   matrix[N_train, D] X_train;
   matrix[N_test, D] X_test;
}

parameters {
   matrix[D, K] beta;
}

model {
 matrix[N_train, K] x_beta = X_train * beta;
 to_vector(beta) ~ normal(0, 2);
 
 for (i in 1:N_train)
 y_train[i] ~ categorical_logit(x_beta[i]');
}

// generated quantities{
//    vector[N_test] y_test;
//    for(i in 1:N_test)
//    y_test[i] = categorical_logit(X_test[i] * beta);
// }