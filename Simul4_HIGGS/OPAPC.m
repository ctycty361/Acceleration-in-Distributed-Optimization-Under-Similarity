function [OPAPC_x, opt_residual, idx, stop_iter] = OPAPC(W, x_init, TOI, mu_mn_reg, L_mx_reg)
global I d eps  
 
L = eye(I)-W;
eigL = eig(L);
chi = eigL(end)/eigL(2);

eig_max = max(eig(L));

%%% Each fi is L smooth and mu scvx.
 

kappa_L = L_mx_reg/mu_mn_reg;


T = floor(sqrt(chi));
c1 = (sqrt(chi)-1)/(sqrt(chi)+1);
tau = min(1, (1+c1^T)/(2*(1-c1^T)*sqrt(kappa_L)));
c2 = (chi+1)/(chi-1);
c3 = 2*chi/((1+chi)*eig_max);
eta = 1/(4*tau*L_mx_reg);
theta = (1+c1^(2*T))/(eta*(1+c1^T)^2);
al = mu_mn_reg;


x0 = x_init;
xf0 = x0;
z0 = zeros(d,I);

opt_residual = {};
opt_residual{1} = evaluate(x0);
idx = {};
idx{1} = 1;

out_iter = 0;
tt = 0;
 
while(opt_residual{out_iter + 1} > eps)
    out_iter = out_iter+1;
    
    xg0 = tau*x0 + (1-tau)*xf0;
    Grad_g = OPAPC_Grad(xg0);
    x_hat = ((1+eta*al)^(-1))*(x0 - eta*(Grad_g-al*xg0+z0));
    z1 = z0 + theta*OPAPC_AccGossip(L, x_hat, T, c2, c3);
    x1 = ((1+eta*al)^(-1))*(x0-eta*(Grad_g - al*xg0 + z1));
    xf1 = xg0 + 2*tau/(2-tau)*(x1-x0);
    
    tt = tt+T;
    
    %%% increase iter num
    xf0 = xf1;
    z0 = z1;
    x0 = x1;
    
    %%% Algorithm Progress %%%
    res = evaluate(x0)
    opt_residual{out_iter+1} = res;
    idx{out_iter+1} = tt+1;
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
 
end

OPAPC_x = x0;
stop_iter = idx{out_iter + 1};
end

function Grad_g  = OPAPC_Grad(x)
global I d Features Labels lambda
Grad_g = zeros(d, I);
for i = 1:I
    Grad_g(:,i) = SmoothHinge_grad(Features{i}, Labels{i}, x(:,i), lambda, zeros(d,1), 0);
end
end

function out = OPAPC_AccGossip(L, x, T, c2, c3)
%%% input x: d cart I, reshape it to be d*I;
%%% input L: d cart d, kron it to be (dI) cart (dI);
sx = size(x);
d = sx(1);
I = sx(2);
flat_dim = I*d;
x = reshape(x, flat_dim,1);
flat_L = kron(L, eye(d));
a0 = 1;
a1 = c2;
a = zeros(T+1,1);
a(1,1) = a0;
a(2,1) = a1;

x0 = x;
x1 = c2*(eye(flat_dim)-c3*flat_L)*x;
X = zeros(flat_dim,T+1);
X(:,1) = x0;
X(:,2) = x1;
for i = 1:T-1
    a(i+2,1) = 2*c2*a(i+1,1)-a(i,1);
    X(:,i+2) = 2*c2*(eye(flat_dim)-c3*flat_L)*X(:,i+1)-X(:,i);
end
result_x = x-X(:,T+1)/a(T+1,1);
out = reshape(result_x, d, I);
end

