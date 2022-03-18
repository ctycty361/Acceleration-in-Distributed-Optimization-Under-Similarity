data_size = 18000;
lambda = 0.1;
data_eps = 1e-8;
filename = ['./results/MNIST/',num2str(data_size),'_reg',num2str(lambda),'_DataEps',num2str(data_eps),'_.mat'];

load(filename);

figure
ToI = 3000;
fz = 30;
fmean = semilogy(OPAPC_idx, OPAPC_res,'d',APMC_idx, APMC_res, 'r',Mudag_idx,Mudag_res,'b', AccEXTRA_idx, AccEXTRA_res,'g', Cata_L_idx, Cata_L_res,'k',...
     Acc_F_idx, Acc_F_res,'c' );
hold on
set(fmean, 'linewidth', 3);
 
legend({ 'OPAPC','APMC','Mudag','AccEXTRA','Acc-SONATA-L','Acc-SONATA-F' },'FontSize',23.5)
 
xlabel({'Communication cost'}, 'FontSize', fz)
ylabel({'Optimality gap'}, 'FontSize', fz)
set(gca,'FontSize',fz)
title(['N = ' num2str(data_size) , '     \lambda= ' num2str(lambda) ])
ylim([1e-4 inf])
xlim([0, ToI])
set(gcf,'position',[0,0,760,760]);