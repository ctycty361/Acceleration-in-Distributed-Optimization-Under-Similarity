function [AccEXTRA_x,  opt_residual, idx, AccEXTRA_hyper] = AccEXTRA(W, x_init, ToI, c_K)
global I d mu_empri L_empri Q lambda
global beta mu_reg

AccEXTRA_hyper = struct();

x0 = x_init;
y0 = x0;
v0 = zeros(d,I);

mu_vec = mu_empri;
L_vec = L_empri;
for i = 1:I
    eig_mn = min(eig(Q(:,:,i)));
    eig_mx = max(eig(Q(:,:,i)));
    if(eig_mx > L_vec)
        L_vec = eig_mx;
    end
    if(eig_mn < mu_vec)
        mu_vec = eig_mn;
    end
end
L = L_vec+lambda;
mu = mu_vec+lambda;

eigW = sort(eig(W));
eig2 = eigW(end-1);
tau = L*(1-eig2)-mu;

be = L;
al = 0.65*(1/L);
% 1/(4*L)

inner_T = ceil(c_K*1/(1-eig2)*log(L/(mu*(1-eig2))));

q  = mu/(mu+tau);
theta = sqrt(q);

tt = 0;
out_iter = 0;
idx = {};
idx{1} = 1;
opt_residual = {};
opt_residual{1} = evaluate(x0);

AccEXTRA_hyper.q = q;
AccEXTRA_hyper.theta = theta;
AccEXTRA_hyper.c_K = c_K;
AccEXTRA_hyper.al = al;
AccEXTRA_hyper.be = be;
AccEXTRA_hyper.tau = tau;

while(tt < ToI)
    out_iter = out_iter + 1;
    
    embedded = true;
    
    [x1, v1, comm_idx, EXTRA_residual] = EXTRA(W, x0, v0, inner_T, y0, tau, al, be, embedded);
    
    
    
    evaluate(x1)
    tt = tt + comm_idx{end};
    
    y1 = x1 + (1-theta)/(1+theta)*(x1-x0);
    
    x0 = x1;
    v0 = v1;
    y0 = y1;
    
    opt_residual{out_iter+1} = evaluate(x0);
    idx{out_iter+1} = tt+1;
    
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    AccEXTRA_res = opt_residual{out_iter+1}
    
end
AccEXTRA_x = x0;
end







