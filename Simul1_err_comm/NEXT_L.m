function [x, y, opt_residual, idx] = NEXT_L(x_init, tau, aux_x, y, W, ToI, embedded)
global I d L_reg chebyshev_num gamma
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
while(tt < ToI)
    out_iter = out_iter+1;
    %%%%%%% local optimization %%%%%%%%%%%%
    
    step_size = 1/(L_reg + tau);
    
    v = x - step_size*y;
    
    %%%%%%%%% consensus %%%%%%%%
    
    tmp_x = gamma*(v-x)+x;
    
    
    x1 = chebyshev3(W, tmp_x, chebyshev_num);
    
    % gradient update
    new_Grad = zeros(d, I);
    for i = 1:I
        new_Grad(:, i) = grad(i, x1(:, i), tau, aux_x(:, i));
    end
    
    
    y = chebyshev3(W, y + new_Grad - Grad, chebyshev_num);
    tt = tt + 2*chebyshev_num;
    
    
    
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
end