function [out_x, out_v, idx, opt_residual] = EXTRA(W, x0, v0, inner_T, y0, tau, al, be, embedded)
global I d

tt = 0;
out_iter = 0;

opt_residual = {};
opt_residual{1} = evaluate(x0);
idx = {};
idx{1} = 1;

while(tt < inner_T)
    out_iter = out_iter + 1;
    
    x1 = zeros(d,I);
    v1 = zeros(d,I);
    for i = 1:I
        temp_x = x0*W;
        x1(:,i) = x0(:,i) - al*( grad(i, x0(:,i), tau, y0(:,i)) + v0(:,i) + be/2*(x0(:,i) - temp_x(:,i)) );
    end
    for i = 1:I
        temp_x1 = x1*W;
        v1(:,i) = v0(:,i) + be/2*( x1(:,i) - temp_x1(:,i));
    end
    
    tt = tt + 2;
    
    x0 = x1;
    v0 = v1;
    
    opt_residual{out_iter + 1} = evaluate(x0);
    if(embedded)
        idx{out_iter+1} = tt;
    else
        idx{out_iter+1} = tt+1;
    end
    
    if(~embedded)
       comm_iter = tt
       EXTRA_res = opt_residual{out_iter + 1}
    end
    
     % check if the algorithm diverges
    if opt_residual{out_iter + 1} > 1e8
        error('Algorithm diverged! Reduce step size.')
    end
     
end
out_x = x0;
out_v = v0;
end