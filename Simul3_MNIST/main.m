Ns = [18000, 30000, 48000, 60000]';
eps0 = 1e-4;
lambda0 = 0.1;
% N = Ns(4,1);
goal = 4;
tic
for i = 1:4
    N = Ns(i,1);
[comms_cov1_iter_N,data_stats] = validate_cov1_iter_N(N, lambda0, eps0, goal);
 
save(['./results/MNIST_simul2_N_',num2str(N),'_reg_',num2str(lambda0),'_goal',num2str(goal),'_.mat'],'comms_cov1_iter_N','data_stats');
end
t = toc