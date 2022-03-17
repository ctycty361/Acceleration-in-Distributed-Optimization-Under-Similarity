function [APMC_x, APMC_residual, APMC_idx, APMC_hyper] = APMC(W, x_init, ToI, cc_K)
global L_reg mu_reg Q lambda I

APMC_hyper = struct();

L_mx = L_reg;
mu_mn = mu_reg;

for i = 1:I
    Q_k = Q(:,:,i);
    eig_Qk = eig(Q_k);
    if(min(eig_Qk)+lambda < mu_mn)
        mu_mn = min(eig_Qk)+lambda;
    end
    if(max(eig_Qk)+lambda > L_mx)
        L_mx = max(eig_Qk)+lambda;
    end
end

eigW = sort(eig(W));
sig2 = eigW(end-1);
eta = (1-sqrt(1-sig2^2))/(1+sqrt(1-sig2^2));

tt = 0;
out_iter = 0;
APMC_idx = {};
APMC_idx{1} = 1;
opt_residual = {};
opt_residual{1} = evaluate(x_init);

x0 = x_init;
x_m1 = x_init;
 
beta0 = 100;
theta = sqrt(mu_mn/L_mx);
stepsize = 1/L_mx; 

APMC_hyper.stepsize = stepsize;
APMC_hyper.cc_K = cc_K;
APMC_hyper.theta = theta;
APMC_hyper.beta0 = beta0;
while(tt < ToI)
    out_iter = out_iter + 1;
    y0 = x0 + (L_mx * theta - mu_mn)*(1 - theta)/((L_mx-mu_mn)*(theta))*(x0-x_m1);
    z0 = y0 - stepsize * APMC_Grad(y0);
    
    ttt = 0;
    
    z00 = z0;
    z0_m1 = z0;
    
    inner_T = ceil((out_iter-1)*cc_K*sqrt(mu_mn/L_mx)/sqrt(1-sig2));
    inner_T = 1;
    while(ttt <= inner_T)
        z01 = (1+eta)*z00*W - eta*z0_m1;    
        z0_m1 = z00;
        z00 = z01;
        ttt = ttt + 1;    
    end
    tt = tt + ttt;
    
    x1 = (L_mx*theta*z0 + beta0*z00)/(L_mx*theta + beta0);
    
    x_m1 = x0;
    x0 = x1;
    
    APMC_res = evaluate(x0)
    opt_residual{out_iter + 1} = APMC_res;
    APMC_idx{out_iter + 1} = tt + 1;
    
end
APMC_x = x0;
APMC_residual = opt_residual;
end

function Grad_g = APMC_Grad(x)
global I d
Grad_g = zeros(d, I);
for i = 1:I
    Grad_g(:,i) = grad(i, x(:,i), 0, zeros(d,1))/I;
end
end
