function [Mudag_x, Mudag_res, Mudag_idx, Mudag_hyper] = Mudag(W, x_init, ToI, c_K )

Mudag_hyper = struct();
global I L_reg mu_reg Q lambda d

L = L_reg;
mu = mu_reg;

kappa_g = L/mu;
mu_mn = mu;
M = L;
for i = 1:I
    Q_k = Q(:,:,i);
    eig_Qk = eig(Q_k);
    if(min(eig_Qk)+lambda < mu_mn)
        mu_mn = min(eig_Qk)+lambda;
    end
    if(max(eig_Qk)+lambda > M)
        M = max(eig_Qk)+lambda;
    end
end

al = sqrt(mu/L);
eta = 1/L;  
 
Mudag_hyper.c_K = c_K;
Mudag_hyper.eta = eta;
Mudag_hyper.al = al;

eigvals = eig(W);
eigvals = sort(eigvals);
eig2 = eigvals(end-1);
K = ceil(c_K*1/(sqrt(1-eig2))*log(M/L*kappa_g));
 
% initial values are required to be consensus
x_INIT = mean(x_init,2);
x_init = kron(ones(1,I),x_INIT);
y_init = x_init;
y_m1 = y_init;
x0 = x_init;
y0 = y_init;

eta_w = (1-sqrt(1-eig2^2))/(1+sqrt(1-eig2^2));

residual = evaluate(x0);
opt_residual = {};
opt_residual{1} = residual;
idx = {};
idx{1} = 1;
tt = 0;
out_iter = 0;
while(tt < ToI)
    out_iter = out_iter + 1;
    
    x_para = y0 + (x0-y_m1) - eta*(Mudag_Grad(y0) - (out_iter > 1)*Mudag_Grad(y_m1));
    x1 = Mudag_FastMix(x_para, K, W, eta_w);
    y1 = x1 + (1-al)/(1+al)*(x1-x0);
    tt = tt + K;

    y_m1 = y0;
    y0 = y1;
    x0 = x1;
    
    idx{out_iter+1} = tt+1;
    opt_residual{out_iter+1} = evaluate(x0);
    
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    Mudag_res = opt_residual{out_iter+1}
 
end

Mudag_x = x0;
Mudag_res = opt_residual;
Mudag_idx = idx;
end

function Grad_g = Mudag_Grad(y)
global d I

Grad_g = zeros(d, I);
for i = 1:I
    Grad_g(:,i) = grad(i, y(:,i), 0, zeros(d,1))/I;
end
end

function out = Mudag_FastMix(x, K, W, eta_w)
x0 = x';
x_m1 = x';
for k = 0:K
    x1 = (1+eta_w)*W*x0 - eta_w*x_m1;
    x_m1 = x0;
    x0 = x1;
end
out = x0';
end

