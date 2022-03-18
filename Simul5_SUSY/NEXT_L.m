function [x0, y, opt_residual, idx, diffs, diff_idx] = NEXT_L(x_init, tau, aux_x, y, W, ToI, embedded, L_reg, outtt_iter)
global I d chebyshev_num gamma lambda Features Labels eps
% initialize
x00 = x_init;
x0 = x_init;
opt_residual = {};
opt_residual{1} = evaluate(x0);

diffs = {};
diff_idx = {};

if ~embedded
    tau = 0;
    aux_x = zeros(d, I);
end

Grad = zeros(d, I);
for i = 1:I
    Grad(:, i) = Logis_grad(Features{i}, Labels{i}, x0(:, i), lambda, aux_x(:, i), tau);
end

tt = 0;
idx = {};
if(~embedded)
    idx{1} = 1;
end
out_iter = 0;

while(tt < ToI)
    
    out_iter = out_iter+1;
    %%%%%%% local optimization %%%%%%%%%%%%
    v = x0 - 1.1/(L_reg + tau) * y;
    
    %%%%%%%%% consensus %%%%%%%%
    
    tmp_x = gamma*(v-x0)+x0;
    x1 = chebyshev3(W, tmp_x, chebyshev_num);
    tt = tt +  chebyshev_num;
    
    if(evaluate(x1) <= eps)
        x0 = x1;
        res = evaluate(x0);
        opt_residual{out_iter + 1} = res;
        if(~embedded)
            idx{out_iter+1} = tt+1;
        else
            idx{out_iter} = tt;
        end
        break;
    end
    
    % gradient update
    new_Grad = zeros(d, I);
    for i = 1:I
        new_Grad(:, i) = Logis_grad(Features{i}, Labels{i}, x1(:, i), lambda, aux_x(:, i), tau);
    end
    y = chebyshev3(W, y + new_Grad - Grad, chebyshev_num);
    tt = tt +  chebyshev_num;
    
    
    
    Grad = new_Grad;
    x0 = x1;
    
    %%%%% alg progression metric %%%%
    diff = norm(x0-x00);
    x00 = x0;
    diffs{out_iter} = diff;
    
    diff_idx{out_iter} = tt;
    if(~embedded)
        idx{out_iter+1} = tt+1;
    else
        idx{out_iter} = tt;
    end
    
    res = evaluate(x0);
    opt_residual{out_iter + 1} = res;
    
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    
    %     report progress once in a while
    if ~embedded && mod(tt, 500) == 0
        fprintf('%d-th round, the error is %f\n', tt, opt_residual{out_iter + 1});
    end
    
end
end