
load('./results_SL_SF_updated/comm_kappa_beta.mat');

% 1, APMC; 2, Mudag; 3, AccEXTR; 4. ASL; 5, ASF
 
plot = 'k'; % b: beta; k: kappa
if(plot == 'k')

%% iter vs kappa %%
kappa_idx = zeros(20,1);

beta_mu = 0;
for i = 1:20
   beta_mu = beta_mu + comms_kappa(9,i)/comms_kappa(8,i); 
end
beta_mu = beta_mu/20;
 
beta = comms_beta(9,1);
for i = 1:20
   kappa_idx(i,1) = comms_kappa(7,i)/comms_kappa(8,i); 
end
% APMC
APMC_iter = zeros(20,1);
for i = 1:20
   APMC_iter(i) = comms_kappa(1,i); 
end

% Mudag
Mudag_iter = zeros(20,1);
for i = 1:20
   Mudag_iter(i) = comms_kappa(2,i); 
end

% AccEXTRA
AccEXTRA_iter = zeros(20,1);
for i = 1:20
   AccEXTRA_iter(i) = comms_kappa(3,i); 
end

% ASL
ASL_iter = zeros(20,1);
for i = 1:20
   ASL_iter(i) = comms_kappa(4,i); 
end

% ASF
ASF_iter = zeros(20,1);
for i = 1:20
   ASF_iter(i) = comms_kappa(5,i); 
end
 

fz = 30; 
fmean = semilogy(kappa_idx, APMC_iter,'r', kappa_idx, Mudag_iter,'b', kappa_idx, AccEXTRA_iter,'g', kappa_idx, ASL_iter,'k', kappa_idx, ASF_iter,'c');

hold on
set(fmean, 'linewidth', 3);
 
legend({'APM-C', 'Mudag','ACC-EXTRA', 'ACC-SONATA-L', 'ACC-SONATA-F'}, 'FontSize', 23.5) 
xlabel({'\kappa'}, 'FontSize', fz)
ylabel({'Number of communications'}, 'FontSize', fz)
xlim([300, 1050])
ylim([10^3 6*10^5])
set(gca,'FontSize',fz)
set(gcf,'position',[0,0,760,760]) 
title(['\beta/\mu \approx ' num2str(beta_mu)  ]);
grid on;
axis square

else
%% iter vs beta %%
beta_idx = zeros(20,1);

kappas = zeros(20,1);
kappa = 0;
for i = 1:20
   kappas(i,1) = comms_beta(7,i)/comms_beta(8,i);
   kappa = kappa + comms_beta(7,i)/comms_beta(8,i); 
end
kappa = kappa/20;

% beta = comms_beta(9,1);
for i = 1:20
   beta_idx(i,1) = comms_beta(9,i)/comms_beta(8,i); 
end

% APMC
APMC_iter = zeros(20,1);
for i = 1:20
   APMC_iter(i) = comms_beta(1,i); 
end

% Mudag
Mudag_iter = zeros(20,1);
for i = 1:20
   Mudag_iter(i) = comms_beta(2,i); 
end

% AccEXTRA
AccEXTRA_iter = zeros(20,1);
for i = 1:20
   AccEXTRA_iter(i) = comms_beta(3,i); 
end

% ASL
ASL_iter = zeros(20,1);
for i = 1:20
   ASL_iter(i) = comms_beta(4,i); 
end

% ASF
ASF_iter = zeros(20,1);
for i = 1:20
   ASF_iter(i) = comms_beta(5,i); 
end

fz = 30;
fmean = semilogy(beta_idx, APMC_iter, 'r',beta_idx, Mudag_iter,'b', beta_idx, AccEXTRA_iter,'g', beta_idx, ASL_iter,'k', beta_idx, ASF_iter,'c'); 

hold on
set(fmean, 'linewidth', 3);
 
legend({'APM-C', 'Mudag','ACC-EXTRA', 'ACC-SONATA-L', 'ACC-SONATA-F'}, 'FontSize', 23.5) 
xlabel({'\beta/\mu'}, 'FontSize', fz)
ylabel({'Number of communications'}, 'FontSize', fz)
xticks(0:200:1000)
ylim([4*10^2 2*10^5])
xlim([0 1100])
set(gca,'FontSize',fz)
set(gcf,'position',[0,0,760,760]) 
title(['\kappa \approx ' num2str(kappa)  ]);
grid on;
% axis square;
end



