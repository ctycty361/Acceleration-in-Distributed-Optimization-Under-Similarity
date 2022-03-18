lambda = 0.01;
Ns = [90000, 150000, 240000, 600000]';
Iter_N  = zeros(4,6);
Mu = zeros(4,1);
for i = 1:4
    N = Ns(i);
    filename =['./results/susy_simul5_N_',num2str(N),'_reg_',num2str(lambda),'_.mat'];
    load(filename);
    Iter_N(i,:) = comms_SUSY_iter_N(1:6,1);
 
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
Ns = [90000, 150000, 240000, 600000]';
% fmean = semilogy(Ns, OPAPC_iter,'d', Ns, APMC_iter, 'r',Ns,Mudag_iter,'b', Ns, AccEXTRA_iter,'g', Ns,ASL_iter,'k',...
%     Ns, ASF_iter,'c');
fmean = semilogy( Ns, OPAPC_iter,'d',Ns, ASF_iter,'c');
set(fmean,'linewidth',3)
fz0 = 23.5;
% legend({ 'APM-C','Mudag','ACC-EXTRA','ACC-SONATA-L','ACC-SONATA-F'},'FontSize',fz0);
legend({'OPAPC','ACC-SONATA-F'}, 'FontSize',fz0)
xlabel({'Total data size'}, 'FontSize', fz)
ylabel({'Number of communications'}, 'FontSize', fz)
 xticks(0:200000:640000)
xlim([0 640000])
title(['     \lambda= ' num2str(lambda)  ])
set(gca,'FontSize',fz)
set(gcf,'position',[0,0,760,760]);


