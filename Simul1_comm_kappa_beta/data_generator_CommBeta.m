function [x_opt, f_opt, mu_regularize, L_regularize ] = data_generator_CommBeta(beta, Q, b, mu_empri, L_empri, raw_b, D)
global n I d lambda


Q_mean = mean(Q,3);
b_mean = mean(b,2);

mu_regularize = mu_empri + lambda;
L_regularize = L_empri + lambda;
% kappa_regularize = L_regularize / mu_regularize;

if beta <= mu_regularize || L_regularize <= 2*mu_regularize 
    error('beta < mu, invalid simulation set up, consider reduce n');
end

% compute the global minimum value f^star
x_opt = (Q_mean + lambda * eye(d)) \ b_mean;
f_opt = 0;
for i = 1 : I
    f_opt = f_opt + norm(D{i} * x_opt - raw_b{i})^2 / (2 * n);
end
f_opt = f_opt / I + lambda * norm(x_opt)^2 / 2;
end