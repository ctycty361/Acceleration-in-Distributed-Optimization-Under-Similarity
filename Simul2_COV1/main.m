rng(10032021,'v4');
Ns = [ 9000, 18000, 42000, 150000]';
eps0 = 1e-4;
lambda0 = 0.01;
N = Ns(4,1);
for i = 1:4
    N = Ns(i,1);
[comms_cov1_iter_N,data_stats] = validate_cov1_iter_N(N, lambda0, eps0);
 
save(['./results2/cov1022_simul2_N_',num2str(N),'_reg_',num2str(lambda0),'_.mat'],'comms_cov1_iter_N','data_stats');
end