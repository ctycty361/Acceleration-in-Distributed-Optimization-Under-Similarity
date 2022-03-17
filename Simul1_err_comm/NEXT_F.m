function [x0, y, opt_residual, idx] = NEXT_F(beta, init_x, tau, aux_x, y, W, ToI, embedded)
global I d Q b lambda chebyshev_num gamma
% initialize
x0 = init_x;
opt_residual = {};
opt_residual{1} = evaluate(x0);

if ~embedded
    tau = 0;
    aux_x = zeros(d, I);
end
Grad = zeros(d, I);
for i = 1:I
    Grad(:, i) = grad(i, x0(:, i), tau, aux_x(:, i));
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
    v = zeros(d, I);
    for i = 1:I
        v(:,i) = (Q(:,:,i) + (lambda + beta + tau) * eye(d)) \ ...
            (b(:,i) + beta * x0(:,i) + tau * aux_x(:, i) + Grad(:, i) - y(:,i));
    end
    
    %%%%%%%%% consensus %%%%%%%%
    
    tmp_x = gamma*(v-x0)+x0;
    
    x1 = chebyshev3(W, tmp_x, chebyshev_num);
    
    
    % gradient update
    new_Grad = zeros(d, I);
    for i = 1:I
        new_Grad(:, i) = grad(i, x1(:, i), tau, aux_x(:, i));
    end
    
    y = chebyshev3(W, y+new_Grad-Grad, chebyshev_num);
    tt = tt+2*chebyshev_num;
    
    
    
    Grad = new_Grad;
    x0 = x1;
    
    %%%%% alg progression metric %%%%
    opt_residual{out_iter + 1} = evaluate(x0);
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