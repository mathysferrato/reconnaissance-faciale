function [mu, Sigma] = estimation_mu_Sigma(X)

    n = size(X, 1);
    x_barre = (1/n) * X' * ones(n, 1);
    X_c = X - ones(n, 1) * x_barre';

    mu = (1/n) * X' * ones(n, 1);
    Sigma = (1/n) * (X_c)' * X_c;

end

