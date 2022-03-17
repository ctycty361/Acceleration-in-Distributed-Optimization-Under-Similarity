function [x, opt_residual, idx, ASF_hyper] = Acc_NEXT_F(W, init_x, ToI, inner_T)
global I d beta mu_reg gamma
% initialize

ASF_hyper = struct();

alpha = sqrt(mu_reg / beta);
x = init_x;
old_x = init_x;
z = init_x;
y = zeros(d, I);
for i = 1:I
    y(:, i) = grad(i, x(:, i), 0, zeros(d, 1));
end
 
opt_residual = {};
opt_residual{1} = evaluate(z);

idx = {};
idx{1} = 1;
tt = 0;
out_iter = 0;
 
tau = beta-mu_reg;

ASF_hyper.alpha = alpha;
ASF_hyper.inner_T = inner_T;
ASF_hyper.tau = tau;
ASF_hyper.gamma = gamma;
while(tt < ToI)
    out_iter = out_iter + 1;
    [new_z, y, residual, comm_idx] = NEXT_F(beta, z, beta-mu_reg, x,...
        y + (beta-mu_reg)*(old_x - x), W, inner_T, true);
    new_x = new_z + (1 - alpha) / (1 + alpha) * (new_z - z);
    
    tt = tt+comm_idx{end};
    
    old_x = x;
    x = new_x;
    z = new_z;
    %%%%% alg progression metric %%%%
    opt_residual{out_iter + 1} = evaluate(z);
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
end