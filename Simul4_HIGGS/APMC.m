function [APMC_x, APMC_residual, APMC_idx, APMC_stopIter] = APMC(W, x_init, ToI, mu_mn, L_mx, cc_K)
global eps
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
theta = sqrt(mu_mn/L_mx);

beta0 = 100;

stepsize = 1/L_mx;

while(eps < opt_residual{out_iter+1})
    out_iter = out_iter + 1;
    y0 = x0 + (L_mx * theta - mu_mn)*(1 - theta)/((L_mx-mu_mn)*(theta))*(x0-x_m1);
    z0 = y0 - stepsize * APMC_Grad(y0);
    
    ttt = 0;
    
    z00 = z0;
    z0_m1 = z0;
    
    inner_T = ceil((out_iter-1)*cc_K*sqrt(mu_mn/L_mx)/sqrt(1-sig2));
    while(ttt <= inner_T)
        z01 = (1+eta)*z00*W - eta*z0_m1;
        
        z0_m1 = z00;
        z00 = z01;
        ttt = ttt + 1;
        
        x_temp = (L_mx*theta*z0 + beta0*z00)/(L_mx*theta + beta0);
        res = evaluate(x_temp);
        if(res <= eps)
            break;
        end
    end
    tt = tt + ttt;
    
    x1 = (L_mx*theta*z0 + beta0*z00)/(L_mx*theta + beta0);
    
    x_m1 = x0;
    x0 = x1;
    
    opt_residual{out_iter + 1} = evaluate(x0);
    APMC_idx{out_iter + 1} = tt + 1;
    
    comm_iter = tt
    APMC_res = opt_residual{out_iter + 1}
    
end
APMC_x = x0;
APMC_residual = opt_residual;
APMC_stopIter = APMC_idx{out_iter + 1}-1;
end

function Grad_g = APMC_Grad(x)
global I d Features Labels lambda
Grad_g = zeros(d, I);
for i = 1:I
    Grad_g(:,i) = SmoothHinge_grad(Features{i}, Labels{i}, x(:,i), lambda, zeros(d,1), 0)/I;
end
end
