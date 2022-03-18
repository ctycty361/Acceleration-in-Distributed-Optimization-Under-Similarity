% lambda = 1;
function data_stats = test_data(N, filename)
 
data_stats = struct();
 
%  filename = ['cov1_data_opt_',num2str(data_size),'_reg_',num2str(lambda),'_.mat'];
load( filename);

data_size = size(U_stack,1);
d = size(U_stack,2); 
 
Hessian = U_stack'*((V_stack.^2).^U_stack);
Hessian = Hessian/data_size;
% load('cov1_Hessian_581012_reg1.mat');
% max_diff = 0;

diff = 0;
I = 30;
Hessians = {};
for i = 1:I
    U_i = U{i};
    V_i = V{i};
    size_local = size(U_i,1);
    Hessian_i = U_i'*((V_i.^2).^U_i)/size_local;
    if(norm(Hessian_i - Hessian) >= diff)
        diff = norm(Hessian_i - Hessian);
    end
    Hessians{i} = Hessian_i;
end

eig_Hessian = sort(eig(Hessian));
L_mean = eig_Hessian(end);
mu_mean = eig_Hessian(1);
L_mx = L_mean;
mu_mn = mu_mean;


for i = 1:I
    eig_local_Hessian = sort(eig(Hessians{i}));
    L_i = eig_local_Hessian(end);
    mu_i = eig_local_Hessian(1);
    if(L_i > L_mx)
        L_mx = L_i;
    end
    if(mu_i < mu_mn)
        mu_mn = mu_i;
    end
end

data_stats.mu_mn = mu_mn;
data_stats.L_mx = L_mx;
data_stats.mu_mean = mu_mean;
data_stats.L_mean = L_mean;
data_stats.diff = diff;
end
 




