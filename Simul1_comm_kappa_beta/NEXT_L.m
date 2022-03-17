function [x, y, opt_residual, idx, stopIter] = NEXT_L(x_init, tau, aux_x, y, W, ToI, embedded)
global I d L_reg chebyshev_num gamma eps
% initialize
x = x_init;
opt_residual = {};
opt_residual{1} = evaluate(x);

if ~embedded
    tau = 0;
    aux_x = zeros(d, I);
end
Grad = zeros(d, I);
for i = 1:I
    Grad(:, i) = grad(i, x(:, i), tau, aux_x(:, i));
end

tt = 0;
idx = {};
if(~embedded)
    idx{1} = 1;
end
out_iter = 0;
while(~((embedded && tt >= ToI) || (~embedded && opt_residual{out_iter + 1} <= eps)) )
    out_iter = out_iter+1;
    %%%%%%% local optimization %%%%%%%%%%%%
    v = x - 1/(L_reg + tau) * y;
    
    %%%%%%%%% consensus %%%%%%%%
    tmp_x = gamma*(v-x)+x;
    
    x1 = chebyshev3(W, tmp_x, chebyshev_num);
    tt = tt +  chebyshev_num;
    
    if(evaluate(x1) <= eps)
        x = x1;
        res = evaluate(x);
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
        new_Grad(:, i) = grad(i, x1(:, i), tau, aux_x(:, i));
    end
    
    y = chebyshev3(W, y + new_Grad - Grad, chebyshev_num);
    tt = tt +  chebyshev_num;
    
    
    
    Grad = new_Grad;
    x = x1;
    
    %%%%% alg progression metric %%%%
    res = evaluate(x);
    opt_residual{out_iter + 1} = res;
    if(~embedded)
        idx{out_iter+1} = tt+1;
    else
        idx{out_iter} = tt;
    end
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    
    
end
stopIter = -1;
if(~embedded)
    stopIter = idx{out_iter+1}-1;
end
end