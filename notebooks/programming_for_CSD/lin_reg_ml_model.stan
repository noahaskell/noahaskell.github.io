data {
  int Nobs; // number of observations per subject
  int Nprd; // number of predictors (including intercept)
  int Nsub; // number of subject
  vector[Nobs] y[Nsub]; // dependent variable
  matrix[Nobs,Nprd] X[Nsub]; // predictor matrices
}
parameters {
  vector[Nprd] beta[Nsub]; // subject-specific coefficients
  vector[Nprd] mu; // means governing betas
  real<lower=0.01> sigma[Nsub]; // data error variance
  real<lower=0.01> tau[Nprd]; // between-subject variation
}
transformed parameters {
  vector[Nobs] y_hat[Nsub]; // predicted values for y
  for(i in 1:Nsub)
    y_hat[i] = X[i] * beta[i];
}
model {
  for(i in 1:Nsub){
    y[i] ~ normal(y_hat[i], sigma[i]);
    for(j in 1:Nprd)
      beta[i,j] ~ normal(mu[j],tau[j]);
  }

  mu ~ normal(0,1);
  tau ~ lognormal(1,2);
  sigma ~ lognormal(1,2);
}
