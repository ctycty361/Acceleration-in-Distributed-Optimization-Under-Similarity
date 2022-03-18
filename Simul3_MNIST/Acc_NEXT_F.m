function [x, opt_residual, idx, stop_iter] = Acc_NEXT_F(W, init_x, ToI, inner_T, local_eps, local_step_size, beta, mu_reg)
global I d Features Labels lambda eps
% initialize

alpha = sqrt(mu_reg / beta);
x = init_x;
old_x = init_x;
z = init_x;
y = zeros(d, I);
for i = 1:I
%     y(:, i) = grad(i, x(:, i), 0, zeros(d, 1));
 
    y(:,i) = SmoothHinge_grad(Features{i}, Labels{i}, x(:,i), lambda, zeros(d,1), 0);
end
% outer_T = floor(ToI / inner_T);
% opt_residual = zeros(1, outer_T+1);
% opt_residual(1) = evaluate(x);
opt_residual = {};
opt_residual{1} = evaluate(z);

idx = {};
idx{1} = 1;
tt = 0;
out_iter = 0;
% for tt = 1:outer_T
% while(tt < ToI)
while(eps < opt_residual{out_iter + 1})
    out_iter = out_iter + 1;
    [new_z, y, residual, comm_idx] = NEXT_F(local_step_size, beta, local_eps, z, beta-mu_reg, x,...
        y + (beta-mu_reg)*(old_x - x), W, inner_T, true);
    new_x = new_z + (1 - alpha) / (1 + alpha) * (new_z - z);
    
    tt = tt+comm_idx{end};
    
    old_x = x;
    x = new_x;
    z = new_z;
    %%%%% alg progression metric %%%%
    opt_residual{out_iter + 1} = evaluate(z);
    idx{out_iter+1} = tt+1;
    
    Acc_SONATA_F_res = opt_residual{out_iter + 1}
    comm_iter = tt
    
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    
    % report progress once in a while
    if mod(tt, 100) == 0
        fprintf('%d-th round, the error is %f\n', tt, opt_residual{out_iter + 1});
    end    
end
stop_iter = idx{out_iter + 1}-1;
end