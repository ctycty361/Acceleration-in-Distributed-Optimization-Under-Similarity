function [comm_nums, L_reg0, mu_reg0, kappa, beta0] = validate_data_change(eps0, lambda0, data_info)
rng(10032021,'v4');
global eps I d n lambda Q b x_opt f_opt mu_reg L_reg beta chebyshev_num mu_empri L_empri  gamma
%%%% Problem parameters %%%%
lambda = lambda0; gamma = 1; eps = eps0;
ToI = 4000;

%%%% Problem data generation %%%%
I = data_info.I;
d = data_info.d;
n = data_info.n;
mu_empri = data_info.mu_empri;
L_empri = data_info.L_empri;
raw_b = data_info.raw_b;
D = data_info.D;
Q = data_info.Q;
b = data_info.b;
beta = data_info.beta;

[x_opt, f_opt, mu_reg, L_reg] = data_generator_CommBeta(beta, Q, b, mu_empri, L_empri, raw_b, D);

fprintf('global cn is %f,   similarity cn is %f\n', L_reg/mu_reg, beta/mu_reg);
kappa = L_reg/mu_reg;
mu_reg0 = mu_reg;
L_reg0 = L_reg;
beta0 = beta;

p = 0.5; %0.9
W = undirected_graph_generator(I,p);
W_PSD = (W+eye(I))/2;

L = eye(I)-W;
eigL = sort(eig(L));
chi = max(eigL)/eigL(2);
chebyshev_num = floor(sqrt(chi));
 
x_init = zeros(d,I);
Grad = zeros(d, I);
for i = 1:I
    Grad(:, i) = grad(i, x_init(:, i), 0, zeros(d, 1));
end

%% algorithms

comm_nums = zeros(5,1);

c_K = 0.2;
[AccEXTRA_x, AccEXTRA_res, AccEXTRA_idx, AccEXTRA_hyper, AccEXTRA_stopIter] = AccEXTRA(W, x_init, ToI,c_K);
AccEXTRA_res = cell2mat(AccEXTRA_res);
AccEXTRA_idx = cell2mat(AccEXTRA_idx);
disp(['AccEXTRA Err: ', num2str(AccEXTRA_res(end))])
comm_nums(3,1) = AccEXTRA_stopIter;

cc_K = 0.01;
[APMC_x, APMC_res, APMC_idx, APMC_hyper, APMC_stopIter] = APMC(W_PSD, x_init, ToI, cc_K);
APMC_res = cell2mat(APMC_res);
APMC_idx = cell2mat(APMC_idx);
disp(['APM-C Err: ', num2str(APMC_res(end))]);
comm_nums(1,1) = APMC_stopIter;

inner_T_AF = ceil(log(beta/mu_reg));
[Acc_F_x, Acc_F_res, Acc_F_idx, ASF_hyper, ASF_stopIter] = Acc_NEXT_F(W, x_init, ToI, inner_T_AF);
Acc_F_res = cell2mat(Acc_F_res);
Acc_F_idx = cell2mat(Acc_F_idx);
disp(['Acc-F Err: ', num2str(Acc_F_res(end))]);
comm_nums(5,1) = ASF_stopIter;

inner_T_CL = ceil(log(L_reg/mu_reg));
[Cata_L_x, Cata_L_res, Cata_L_idx, ASL_hyper, ASL_stopIter] = Cata_NEXT_L(W, x_init, ToI, inner_T_CL);
Cata_L_res = cell2mat(Cata_L_res);
Cata_L_idx = cell2mat(Cata_L_idx);
disp(['Cata-L Err: ', num2str(Cata_L_res(end))]);
comm_nums(4,1) = ASL_stopIter;

c_K = 0.2;
[Mudag_x, Mudag_res, Mudag_idx, Mudag_hyper, Mudag_stopIter] = Mudag(W_PSD, x_init, ToI, c_K ); % The best c_k for Mudag is 0.24
Mudag_res = cell2mat(Mudag_res);
Mudag_idx = cell2mat(Mudag_idx);
disp(['Mudag Err: ', num2str(Mudag_res(end))]);
comm_nums(2,1) = Mudag_stopIter;
 
end

 