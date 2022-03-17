function [x, opt_residual, idx, ASL_hyper, ASL_stopIter] = Cata_NEXT_L(W, init_x, ToI, inner_T)
global I d mu_reg L_reg gamma eps
% initialize

ASL_hyper = struct();
 
alpha = sqrt(mu_reg / L_reg);
x = init_x;
old_x = init_x;
z = init_x;
y = zeros(d, I);
for i = 1:I
    y(:, i) = grad(i, x(:, i), 0, zeros(d, 1));
end
tau = L_reg-mu_reg;
out_iter = 0;

opt_residual = {};
opt_residual{1} = evaluate(z);

tt = 0;
idx = {};
idx{1} = 1;

ASL_hyper.tau = tau;
ASL_hyper.alpha = alpha;
ASL_hyper.inner_T = inner_T;
ASL_hyper.gamma = gamma;
% while(tt < ToI)
while(eps < opt_residual{out_iter + 1})
    out_iter = out_iter + 1;
    if(out_iter == 40)
        'f'
    end
    [new_z, y, residual, comm_idx,~] = NEXT_L(z, tau, x, y + tau*(old_x - x), W, inner_T, true);
    new_x = new_z + (1 - alpha) / (1 + alpha) * (new_z - z);
    
    tt = tt+comm_idx{end};
    
    old_x = x;
    x = new_x;
    z = new_z;
    %%%%% alg progression metric %%%%
    AccSONATA_L_res = evaluate(z);
    opt_residual{out_iter + 1} = AccSONATA_L_res;
    idx{out_iter+1} = tt+1;
    
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    
    % report progress once in a while
    if mod(tt, 100) == 0
        fprintf('%d-th round, the error is %f\n', tt, opt_residual{out_iter + 1});
    end
end
ASL_stopIter = idx{out_iter + 1}-1;
end