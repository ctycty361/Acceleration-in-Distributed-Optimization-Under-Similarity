Ns = [120000, 240000, 480000, 900000]';

eps0 = 1e-4;
lambda0 =0.05;

tic
for i = [2,1,3,4]
    N = Ns(i,1);
    data_filename = ['HIGGS_',num2str(N),'.mat'];
    if(~exist(data_filename))
        tic
        [HIGGS_labels, HIGGS_features] = libsvmread('./data/HIGGS'); 
        % The dataset HIGGS can be downloaded from 
        % https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
        t1 = toc
        HIGGS(N, HIGGS_features, HIGGS_labels);
    end
    [comms_cov1_iter_N,data_stats] = validate_cov1_iter_N(N, lambda0, eps0, data_filename);
    save(['./results4/HIGGS_simul2_N_',num2str(N),'_reg_',num2str(lambda0),'_.mat'],'comms_cov1_iter_N','data_stats');
end
t = toc