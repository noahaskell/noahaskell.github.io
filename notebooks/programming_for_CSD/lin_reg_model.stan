data {
  int Nobs; // number of observations
  int Nprd; // number of predictors (including intercept)
  vector[Nobs] y; // dependent variable
  matrix[Nobs,Nprd] X; // predictor matrix
}
parameters {
  vector[Nprd] beta; // coefficients
  real<lower=0.01> sigma; // error variance
}
transformed parameters {
  vector[Nobs] y_hat; // predicted value for y
  y_hat = X * beta;
}
model {
  y ~ normal(y_hat, sigma);

  beta ~ normal(0,1);
  sigma ~ lognormal(1,2);
}
