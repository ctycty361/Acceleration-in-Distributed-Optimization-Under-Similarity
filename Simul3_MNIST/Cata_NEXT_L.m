function [x, opt_residual, idx, ASL_stopIter] = Cata_NEXT_L(W, init_x, ToI, inner_T, L_reg, mu_reg, tau)
global I d  Features Labels lambda eps
% initialize
alpha = sqrt(mu_reg / (L_reg-mu_reg));
x = init_x;
old_x = init_x;
z = init_x;
y = zeros(d, I);
for i = 1:I
    U_i = Features{i};
    V_i = Labels{i};
    y(:,i) = SmoothHinge_grad(U_i, V_i, x(:,i), lambda, zeros(d,1), 0);
end

out_iter = 0;

opt_residual = {};
opt_residual{1} = evaluate(z);

tt = 0;
idx = {};
idx{1} = 1;
while(eps < opt_residual{out_iter+1})
    out_iter = out_iter + 1;
    
    [new_z, y, residual, comm_idx, diffs, diff_idx] = NEXT_L(z, tau, x, y + tau*(old_x - x), W, inner_T, true, L_reg);
    new_x = new_z + (1 - alpha) / (1 + alpha) * (new_z - z);
    
    tt = tt+comm_idx{end};
    
    old_x = x;
    x = new_x;
    z = new_z;
    %%%%% alg progression metric %%%%
    AccSONATA_L_res = evaluate(z);
    opt_residual{out_iter + 1} = AccSONATA_L_res;
    idx{out_iter+1} = tt+1;
    display(['ASL comm: ', num2str(tt),' res: ',num2str(AccSONATA_L_res)]);
    
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