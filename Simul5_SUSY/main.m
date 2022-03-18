Ns = [90000, 150000, 240000, 600000]';
eps0 = 1e-4;
lambda0 =0.01;
% N = Ns(4,1);
tic
for i = 1:4
    N = Ns(i,1);
[comms_SUSY_iter_N,data_stats] = validate_cov1_iter_N(N, lambda0, eps0);
 
save(['./results/cov1_simul4_N_',num2str(N),'_reg_',num2str(lambda0),'_.mat'],'comms_SUSY_iter_N','data_stats');
end
t = toc