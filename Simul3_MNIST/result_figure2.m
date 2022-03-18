Ns = [18000, 30000, 48000, 60000]';
lambda0 = 0.1;

goal = 4;
Iter_N  = zeros(4,6);
Mu = zeros(4,1);
for i = 1:4
    N = Ns(i);
    filename =['./results/MNIST_simul2_N_',num2str(N),'_reg_',num2str(lambda0),'_goal',num2str(goal),'_.mat'];
    load(filename);
    Iter_N(i,:) = comms_cov1_iter_N;
 
end

APMC_iter = zeros(4,1);
Mudag_iter = zeros(4,1);
AccEXTRA_iter = zeros(4,1);
ASL_iter = zeros(4,1);
ASF_iter = zeros(4,1);
OPAPC_iter = zeros(4,1);

for i = 1:4
    APMC_iter(i,1) =  Iter_N(i,2);
    Mudag_iter(i,1) = Iter_N(i,3);
    AccEXTRA_iter(i,1) = Iter_N(i,4);
    ASL_iter(i,1) = Iter_N(i,5);
    ASF_iter(i,1) = Iter_N(i,6);
 OPAPC_iter(i,1) = Iter_N(i,1);
end
hold on;
fz = 30;
% fmean = semilogy(Ns, OPAPC_iter ,'d', Ns, APMC_iter, 'r',Ns,Mudag_iter,'b', Ns, AccEXTRA_iter,'g', Ns,ASL_iter,'k',...
%     Ns, ASF_iter,'c' );
fmean = semilogy(Ns, OPAPC_iter,'d', Ns, ASF_iter,'c' );
set(fmean,'linewidth',3)
fz0 = 23.5;
legend({'OPAPC','ACC-SONATA-F'},'FontSize',fz0);
% legend({ 'APM-C','Mudag','ACC-EXTRA','ACC-SONATA-L','ACC-SONATA-F'},'FontSize',fz0);
xlabel({'Total data size'}, 'FontSize', fz)
ylabel({'Number of communications'}, 'FontSize', fz)
% xticks(15000:15000:60000)
% xlim([15000 62000])
xticks(10000:10000:70000)
xlim([12000 64000])
title(['     \lambda= ' num2str(lambda0)  ])
set(gca,'FontSize',fz)
set(gcf,'position',[0,0,760,760]);
% axis square;


