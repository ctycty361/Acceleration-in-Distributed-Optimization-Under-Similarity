function [x_opt, f_opt, Q, b, beta, mu_regularize, L_regularize, mu_empri, L_empri, D, raw_b] = data_generator(mu, L)
global n I d lambda

x0 = 5 + randn(d,1);
eig_vec =[mu mu+rand(1,d-2)*(L-mu) L];      % eigen values of population
U = randU(d);                               % random unitary matrix
Sigma = U*diag(eig_vec)*U';                 % covariance of D_i matrices

% generate Ai and bi for each agent
D = cell(1, I);
Q = zeros(d, d, I);
raw_b = cell(1, I);
b = zeros(d, I);
for i = 1: I
    D{i} = mvnrnd(zeros(d,1), Sigma, n);
    raw_b{i} = D{i} * x0 + 0.1 * randn(n,1);
    Q(:,:,i) = D{i}' * D{i} / n;
    b(:,i) = D{i}' * raw_b{i} / n;
end

% compute the empirical kappa
Q_mean = mean(Q, 3);
b_mean = mean(b, 2);
eig_vals = eig(Q_mean);
mu_empri = min(eig_vals);
L_empri = max(eig_vals);
tmp = num2cell(Q, [1,2]);
Hessian_tilde = blkdiag(tmp{:});           % Hessian of extended F
beta=norm(Hessian_tilde - kron(eye(I), Q_mean));  % hessian discrepancy beta

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