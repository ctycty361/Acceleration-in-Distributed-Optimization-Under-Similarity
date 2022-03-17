clear;
rng(10032021,'v4');
eps = 1e-4;
mu0 = 1;
L0 = 1000;
C0 = 100;   % minimum sample size n;
lambda = 0.1;
n0 = 2000; % baseline n0, and leading beta0 for iter vs kappa

%% Fix kappa, change n to change beta %%
% 1, APMC; 2, Mudag; 3, AccEXTR; 4. ASL; 5, ASF
% beta ~ 1/sqrt(n)
tic

I = 30;
d = 40;
n_collection = floor(logspace(4.6,2,20))';
comms_beta = zeros(9,20);

for i = 1:20
    i
    n = n_collection(i,1);
    [mu_empri, L_empri, Q, b, raw_b, D, beta] = data_generator_aux(mu0, L0, I, d, n_collection(i,1));
    
    data_info = struct();
    data_info.I = I;
    data_info.d = d;
    data_info.mu_empri = mu_empri;
    data_info.L_empri = L_empri;
    data_info.raw_b = raw_b;
    data_info.D = D;
    data_info.Q = Q;
    data_info.b = b;
    data_info.beta = beta;
    data_info.n = n_collection(i,1);
    
    [comm_nums, L_reg, mu_reg, kappa, beta] = validate_data_change(eps, lambda, data_info);
    comms_beta(1:5,i) = comm_nums;
    comms_beta(6:9,i) = [lambda, L_reg, mu_reg, beta];
    save('./results_SL_SF_updated/comms_beta.mat','comms_beta','n_collection')
end


%% Fix beta, change lambda to change kappa %%

% keep beta/mu \approx 200;

I = 30;
d = 40;
% n = 2000;   % n 2000 ~ beta 207, % to make mu_empri close to 1, n needs to be larger than 400, treat mu_empri as 1;

lower_kappa = (L0+mu0*sqrt(n0/C0))/(mu0 + mu0*sqrt(n0/C0));

kappa_regularize = logspace(log10(lower_kappa),3,20);
lambda_vals =  0.5*(L0- kappa_regularize .* mu0) ./ (kappa_regularize - 1);
lambda_vals = lambda_vals';

n4kappa = zeros(20,1);
for i = 1:20
    new_mu = lambda_vals(i)+mu0;
    scale = new_mu/mu0;
    n4kappa(i,1) = ceil(n0/scale^2);
end

comms_kappa = zeros(10,20);

for i = 1:20
        i
    n = n4kappa(i)
    lambda = lambda_vals(i,1)
    
    [mu_empri, L_empri, Q, b, raw_b, D, beta] = data_generator_aux(mu0, L0, I, d, n4kappa(i));
    %     beta_div_mu = beta/mu_empri; % we need to keep this fixed
    
    data_info = struct();
    data_info.I = I;
    data_info.d = d;
    data_info.mu_empri = mu_empri;
    data_info.L_empri = L_empri;
    data_info.raw_b = raw_b;
    data_info.D = D;
    data_info.Q = Q;
    data_info.b = b;
    data_info.beta = beta;
    data_info.n = n4kappa(i);
    
    [comm_nums, L_reg, mu_reg, kappa, beta] = validate_data_change(eps, lambda,  data_info);
    comms_kappa(1:5,i) = comm_nums;
    comms_kappa(6:10,i) = [lambda, L_reg, mu_reg, beta, n4kappa(i)];
    save('./results_SL_SF_updated/comms_kappa.mat','comms_kappa','lambda_vals')
end

t = toc;
save('./results_SL_SF_updated/comm_kappa_beta.mat','t','comms_kappa','comms_beta','lambda_vals','n_collection');


function [mu_empri, L_empri, Q, b, raw_b, D, beta] = data_generator_aux(mu, L, I, d, n)
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

eig_vals = eig(Q_mean);
mu_empri = min(eig_vals);
L_empri = max(eig_vals);
tmp = num2cell(Q, [1,2]);
Hessian_tilde = blkdiag(tmp{:});           % Hessian of extended F
beta=norm(Hessian_tilde - kron(eye(I), Q_mean));  % hessian discrepancy beta
end
