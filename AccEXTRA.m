function [AccEXTRA_x,  opt_residual, idx, stop_iter] = AccEXTRA(W, x_init, ToI, mu_vec, L_vec, c_K)
global I d lambda Features Labels eps data_size
 
x0 = x_init;
y0 = x0;
v0 = zeros(d,I);

L = L_vec+lambda;
mu = mu_vec+lambda;

eigW = sort(eig(W));
eig2 = eigW(end-1);
tau = L*(1-eig2)-mu;

be = L_vec;
if(data_size < 40000)
   al =   0.75/be;
else
    al = 0.7/be;
end
 
inner_T = ceil(c_K*1/(1-eig2)*log(L/(mu*(1-eig2))));

q  = mu/(mu+tau);
theta = sqrt(q);

tt = 0;
out_iter = 0;
idx = {};
idx{1} = 1;
opt_residual = {};
opt_residual{1} = evaluate(x0);
 
while(eps < opt_residual{out_iter + 1})
    out_iter = out_iter + 1;
    
    embedded = true;
    
    [x1, v1, comm_idx, EXTRA_residual] = EXTRA(W, x0, v0, inner_T, y0, tau, al, be, embedded);

    
    tt = tt + comm_idx{end};
     
    y1 = x1 + (1-theta)/(1+theta)*(x1-x0);

    x0 = x1;
    v0 = v1;
    y0 = y1;
    
    opt_residual{out_iter+1} = evaluate(x0);
    idx{out_iter+1} = tt+1;
        
    comm_iter = tt
    AccEXTRA_res = opt_residual{out_iter+1}
    
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    comm_iter = tt
    AccEXTRA_res = opt_residual{out_iter+1}
 
end
AccEXTRA_x = x0;
stop_iter = idx{out_iter+1}-1;
end

 




