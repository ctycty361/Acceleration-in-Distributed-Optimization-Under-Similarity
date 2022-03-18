function [comms_cov1_iter_N,data_stats] = validate_cov1_iter_N(N, lambda0, eps0)

comms_cov1_iter_N = zeros(5,1);

global I data_size lambda eps
data_size = N;
lambda = lambda0;
eps = eps0;
I = 30;   
 
filename = ['cov1_data_opt_',num2str(data_size),'_reg_',num2str(lambda),'_.mat'];
stepsize = 0.05;
K = 80000;
if(~exist(filename))
    cov_data_generator(data_size, I, K, stepsize, lambda, filename);
end
load(filename);
data_stats = test_data(data_size, filename); 
%%%% Problem parameters %%%%

global d local_sample chebyshev_num   Features Labels gamma x_opt 
local_sample = local_sampleSize;
d = feature_nums;
ToI = 3000;
Features = U;
Labels = V;
gamma = 1;
x_opt = SH_Xopt;
 

%%%% Problem data generation %%%% 
 
p = 0.5;  
W = undirected_graph_generator(I,p);
W_PSD = (W+eye(I))/2;

L_hat = eye(I)-W;
eigtmp = sort(eig(L_hat));
L = 2/(eigtmp(2)+eigtmp(end))*L_hat;
eigL = sort(eig(L));
eig2 = eigL(2);
eig_max = eigL(end);
chebyshev_num = floor(1/sqrt(eig2/eig_max)); 
 
x_init = zeros(d,I);
Grad = zeros(d, I);
for i = 1:I
    tau = 0;
    x_aux = zeros(d,1);
    local_U = Features{i};
    local_V = Labels{i};
    Grad(:, i) = SmoothHinge_grad(local_U, local_V, x_init(:,i), lambda, x_aux, tau);
end

%% algorithms

%%% general parameters %%%
mu_mn = data_stats.mu_mn;
L_mx = data_stats.L_mx;
mu_mean_= data_stats.mu_mean;
L_mean = data_stats.L_mean;

 
mu_empri_mn = mu_mn;
L_empri_mx = L_mx;
c_K = 0.2;  
[AccEXTRA_x, AccEXTRA_res, AccEXTRA_idx, AccEXTRA_stopIter] = AccEXTRA(W, x_init, ToI, mu_empri_mn, L_empri_mx, c_K);
AccEXTRA_res = cell2mat(AccEXTRA_res);
AccEXTRA_idx = cell2mat(AccEXTRA_idx);
disp(['AccEXTRA Err: ', num2str(AccEXTRA_res(end))])
comms_cov1_iter_N(3,1) = AccEXTRA_stopIter;
 
mu_mn_reg = mu_mn + lambda;
L_mx_reg = L_mx + lambda;
cc_K = 0.01;
[APMC_x, APMC_res, APMC_idx, APMC_stopIter] = APMC(W_PSD, x_init, ToI, mu_mn_reg, L_mx_reg, cc_K);
APMC_res = cell2mat(APMC_res);
APMC_idx = cell2mat(APMC_idx);
disp(['APMC Err: ',num2str(APMC_res(end))])
comms_cov1_iter_N(1,1) = APMC_stopIter;

 
mu_reg_Mudag = data_stats.mu_mean + lambda; 
L_reg_Mudag = L_mean + lambda;   
M = L_mx + lambda;
mu_mn_reg = mu_mn + lambda;
c_K = 0.2;
[Mudag_x, Mudag_res, Mudag_idx, Mudag_stopIter] = Mudag(W_PSD, x_init, ToI, L_reg_Mudag, mu_reg_Mudag, M, mu_mn_reg, c_K); % The best c_k for Mudag is 0.24
Mudag_res = cell2mat(Mudag_res);
Mudag_idx = cell2mat(Mudag_idx);
disp(['Mudag Err: ', num2str(Mudag_res(end))]); 
comms_cov1_iter_N(2,1) = Mudag_stopIter;

 
L_reg_ASL = L_mean + lambda;
mu_reg_ASL = data_stats.mu_mean + lambda;
c_sl = 1;
inner_T_CL = ceil(c_sl*log(L_reg_ASL/mu_reg_ASL));   
tau = L_reg_ASL-2*mu_reg_ASL;
[Cata_L_x, Cata_L_res, Cata_L_idx, Cata_L_stopIter] = Cata_NEXT_L(W, x_init, ToI, inner_T_CL, L_reg_ASL, mu_reg_ASL, tau);
Cata_L_res = cell2mat(Cata_L_res);
Cata_L_idx = cell2mat(Cata_L_idx);
disp(['Cata-L Err: ', num2str(Cata_L_res(end))]);
comms_cov1_iter_N(4,1) = Cata_L_stopIter;
 
mu_reg = data_stats.mu_mean + lambda;
beta = data_stats.diff;
local_step_size = 0.03;
local_eps = 1e-10;
inner_T_AF = ceil(log(beta/mu_reg));
[Acc_F_x, Acc_F_res, Acc_F_idx, Acc_F_stopIter] = Acc_NEXT_F(W, x_init, ToI, inner_T_AF, local_eps, local_step_size, beta, mu_reg);
Acc_F_res = cell2mat(Acc_F_res);
Acc_F_idx = cell2mat(Acc_F_idx);
disp(['Acc-F Err: ', num2str(Acc_F_res(end))]);
comms_cov1_iter_N(5,1) = Acc_F_stopIter;

save(['./results2/cov1_N',num2str(data_size),'_reg',num2str(lambda),'_K',num2str(K),'_.mat'], 'data_stats', ...
    'APMC_idx', 'APMC_res', 'Mudag_idx', 'Mudag_res', ...
    'AccEXTRA_idx', 'AccEXTRA_res', 'Cata_L_idx', 'Cata_L_res','Acc_F_idx','Acc_F_res');

iterations = 1 : (ToI+1);
figure
fz = 20;
fmean = semilogy(APMC_idx, APMC_res, 'r',Mudag_idx,Mudag_res,'b', AccEXTRA_idx, AccEXTRA_res,'g', Cata_L_idx, Cata_L_res,'k',...
     Acc_F_idx, Acc_F_res,'c');
hold on
set(fmean, 'linewidth', 3); 
legend({ 'APMC','Mudag','AccEXTRA','Acc-SONATA-L','Acc-SONATA-F'},'FontSize',fz) 
xlabel({'communication cost'}, 'FontSize', fz)
ylabel({'optimality gap'}, 'FontSize', fz)
set(gca,'FontSize',fz)
title(['n = ' num2str(local_sample) , '     \lambda= ' num2str(lambda) ',     I= ' num2str(I) ',     d = ' num2str(d)])
ylim([1e-4 inf])
xlim([0, ToI])
end