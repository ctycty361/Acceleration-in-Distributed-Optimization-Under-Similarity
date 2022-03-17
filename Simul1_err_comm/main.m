clear;
rng(10032021,'v4');

global I d n lambda Q b x_opt f_opt mu_reg L_reg beta chebyshev_num mu_empri L_empri D raw_b gamma
%%%% Problem parameters %%%%
I = 30;        
d = 40;
n = 1000;  
lambda = 0.1;
gamma = 1;
ToI =3000;

%%%% Problem data generation %%%%
mu0 = 1e0;
L0 = 1000;
[x_opt, f_opt, Q, b, beta, mu_reg, L_reg, mu_empri, L_empri, D, raw_b] = data_generator(mu0, L0);

left = (L0 + sqrt(n/100))/(1+sqrt(n/100));
right =  beta;

fprintf('global cn is %f,   similarity cn is %f\n', L_reg/mu_reg, beta/mu_reg);
p = 0.5;  

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

c_K = 0.2; 
[AccEXTRA_x, AccEXTRA_res, AccEXTRA_idx, AccEXTRA_hyper] = AccEXTRA(W, x_init, ToI,c_K);
AccEXTRA_res = cell2mat(AccEXTRA_res);
AccEXTRA_idx = cell2mat(AccEXTRA_idx);
disp(['AccEXTRA Err: ', num2str(AccEXTRA_res(end))])

c_K = 0.2;
[Mudag_x, Mudag_res, Mudag_idx, Mudag_hyper] = Mudag(W_PSD, x_init, ToI, c_K ); % The best c_k for Mudag is 0.24
Mudag_res = cell2mat(Mudag_res);
Mudag_idx = cell2mat(Mudag_idx);
disp(['Mudag Err: ', num2str(Mudag_res(end))]);

cc_K = 0.01;
[APMC_x, APMC_res, APMC_idx, APMC_hyper] = APMC(W_PSD, x_init, ToI, cc_K);
APMC_res = cell2mat(APMC_res);
APMC_idx = cell2mat(APMC_idx);
disp(['APM-C Err: ', num2str(APMC_res(end))]);

c_sl = 1;
inner_T_CL = ceil(c_sl*log(L_reg/mu_reg));
[Cata_L_x, Cata_L_res, Cata_L_idx, ASL_hyper] = Cata_NEXT_L(W, x_init, ToI, inner_T_CL);
Cata_L_res = cell2mat(Cata_L_res);
Cata_L_idx = cell2mat(Cata_L_idx);
disp(['Cata-L Err: ', num2str(Cata_L_res(end))]);


inner_T_AF = ceil(log(beta/mu_reg));
[Acc_F_x, Acc_F_res, Acc_F_idx, ASF_hyper] = Acc_NEXT_F(W, x_init, ToI, inner_T_AF);
Acc_F_res = cell2mat(Acc_F_res);
Acc_F_idx = cell2mat(Acc_F_idx);
disp(['Acc-F Err: ', num2str(Acc_F_res(end))]);

 
%% plot
iterations = 1 : (ToI+1);
figure
fz = 30;
 
fmean = semilogy(APMC_idx, APMC_res,'r', Mudag_idx, Mudag_res, 'b',AccEXTRA_idx, AccEXTRA_res,'g', Cata_L_idx, Cata_L_res,'k', Acc_F_idx, Acc_F_res,'c');
 
hold on
set(fmean, 'linewidth', 3);
 
fz0 = 23.5;
legend({'APM-C', 'Mudag', 'ACC-EXTRA','ACC-SONATA-L', 'ACC-SONATA-F'}, 'FontSize', fz0)
 
xlabel({'Number of communications'}, 'FontSize', fz)
xticks(0:600:3000);
ylabel({'Optimality gap'}, 'FontSize', fz)
set(gca,'FontSize',fz)
set(gcf,'position',[0,0,760,760])
 
title(['\kappa = ' num2str(L_reg/mu_reg) ',     \beta/\mu = ' num2str(beta/mu_reg)]);
ylim([1e-4 inf])
xlim([0, ToI])
grid on;
axis square
