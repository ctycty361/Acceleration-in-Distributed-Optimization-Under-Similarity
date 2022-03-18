 
Ns = [9000, 18000, 42000, 150000]';
lambda0 = 0.01;

Iter_N  = zeros(4,5);
Mu = zeros(4,1);
for i = 1:4
    N = Ns(i);
 
    filename =['./results2/cov1022_simul2_N_',num2str(N),'_reg_',num2str(lambda0),'_.mat'];
load(filename);
    Iter_N(i,:) = comms_cov1_iter_N;
    
end

APMC_iter = zeros(4,1);
Mudag_iter = zeros(4,1);
AccEXTRA_iter = zeros(4,1);
ASL_iter = zeros(4,1);
ASF_iter = zeros(4,1);
for i = 1:4
    APMC_iter(i,1) =  Iter_N(i,1);
    Mudag_iter(i,1) = Iter_N(i,2);
    AccEXTRA_iter(i,1) = Iter_N(i,3);
    ASL_iter(i,1) = Iter_N(i,4);
    ASF_iter(i,1) = Iter_N(i,5);
end

fz = 30;
fmean = semilogy(Ns, APMC_iter, 'r',Ns,Mudag_iter,'b', Ns, AccEXTRA_iter,'g', Ns,ASL_iter,'k',...
    Ns, ASF_iter,'c');
set(fmean,'linewidth',3)
fz0 = 23.5;
legend({ 'APM-C','Mudag','ACC-EXTRA','ACC-SONATA-L','ACC-SONATA-F'},'FontSize',fz0);
xlabel({'Total data size'}, 'FontSize', fz)
ylabel({'Number of communications'}, 'FontSize', fz)
xlim([0 180000])
title(['     \lambda= ' num2str(lambda0)  ])
set(gca,'FontSize',fz)
set(gcf,'position',[0,0,760,760]);
grid on;
% axis square;


