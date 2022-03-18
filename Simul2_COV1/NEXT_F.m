function [x0, y, opt_residual, idx] = NEXT_F(local_step_size, beta, local_eps, init_x, tau, aux_x, y, W, ToI, embedded)
global I d lambda chebyshev_num gamma Features Labels eps
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
    Grad(:, i) = SmoothHinge_grad(Features{i}, Labels{i}, x0(:,i), lambda, aux_x(:,i), tau);
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
        x_local_init = x0(:,i);
        SH_local_grad = Grad(:,i);
        v(:,i) = SH_localSONATA_Solver(x_local_init, y(:,i)-SH_local_grad, Features{i}, Labels{i}, lambda, aux_x(:,i), tau, beta, local_eps, local_step_size);
    end
    
    %%%%%%%%% consensus %%%%%%%%
    
    tmp_x = gamma*(v-x0)+x0;
    x1 = chebyshev3(W, tmp_x, chebyshev_num);
    tt = tt+chebyshev_num;
    if(evaluate(x1) <= eps)
        x0 = x1;
        if(~embedded)
            idx{out_iter+1} = tt+1;
        else
            idx{out_iter} = tt;
        end
        break
    end
    
    % gradient update
    new_Grad = zeros(d, I);
    for i = 1:I
        new_Grad(:,i) = SmoothHinge_grad(Features{i}, Labels{i}, x1(:,i), lambda, aux_x(:,i), tau);
    end
    y = chebyshev3(W, y+new_Grad-Grad, chebyshev_num);
    tt = tt+ chebyshev_num;
    
    
    
    Grad = new_Grad;
    x0 = x1;
    
    %%%%% alg progression metric %%%%
    opt_residual{out_iter + 1} = evaluate(x0);
    if(~embedded)
        idx{out_iter+1} = tt+1;
    else
        idx{out_iter} = tt;
    end
    
    if(~embedded)
        comm_iter = idx{out_iter}
        SONATA_F_res = opt_residual{out_iter + 1}
    end
    % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
    
end
end