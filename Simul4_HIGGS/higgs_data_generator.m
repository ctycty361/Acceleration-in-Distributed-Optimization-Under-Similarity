function higgs_data_generator(data_size, I, K, step_size, lambda, data_filename, filename,  data_eps)
 
tic
rng(112019,'v4');


 
load(data_filename);
 
U_stack = HIGGS_features1;
V_stack = HIGGS_labels1;
 
[num_instances, feature_nums] = size(HIGGS_features1);
 
V_stack(V_stack ~= 1) = -1;

%%% distribute data evenly %%%
U = cell(1, I);
V = cell(1, I);
row = floor(data_size/I);
local_sampleSize = row;
for i = 1:I-1
    U{i} = U_stack((i-1)*row+1: i*row, :);
    V{i} = V_stack((i-1)*row+1: i*row);
end
U{I} = U_stack((I-1)*row+1:end, :);
V{I} = V_stack((I-1)*row+1:end);

%%%%% minimizing by centralized gradient descent %%%%%
d = feature_nums;

x = zeros(d,1);
 
gradnorm = {};
 
SH_Fdesc = {};
SH_Fvalue = {};

x_aux = zeros(d,1);
tau = 0;
 
iter = 0;
grad_offset = 1;
while(grad_offset > data_eps)
 
    iter = iter + 1;
     G = zeros(d,I);
    for j = 1:I
        G(:,j) = SmoothHinge_grad(U{j}, V{j}, x, lambda,  x_aux, tau);
    end
    g = mean(G,2);
    
    x           =  x - step_size * g;
    gradnorm{iter} = norm(g);
    grad_offset = gradnorm{iter};
 
    F = zeros(d,I);
    for j = 1:I
        F(:,j) = SmoothHinge_F(U{j},V{j}, x, lambda, x_aux, tau);
    end
    SH_Fvalue{iter} = mean(F,2);
    
    if iter > 1
 
        SH_Fdesc{iter-1} = abs(SH_Fvalue{iter-1} - SH_Fvalue{iter});
    end
    timegap = 1;
    if mod(iter, timegap) == 0
        fprintf('gradient descent: %d-th iteration, the error is %f\n', iter, gradnorm{iter});
    end
end

% SH_Fopt = SmoothHinge_Fvalue(K);
SH_Fopt = SH_Fvalue{end};
SH_Xopt = x;

%%%%% plot %%%%%
figure;

subplot(1,3,1);
semilogy(1:iter, cell2mat(gradnorm));
xlabel('Iteration')
ylabel('SH-Gradient Norm')

subplot(1,3,2);
semilogy(1:iter, cell2mat(SH_Fvalue));
xlabel('Iteration')
ylabel('SH-F value')

subplot(1,3,3);
semilogy(1:iter-1, cell2mat(SH_Fdesc));
xlabel('Iteration')
ylabel('descent per step of SH_F value')
t = toc
 
save(filename, 'lambda','V_stack','U_stack','U', 'V', 'SH_Fopt', 'SH_Xopt', 'SH_Fvalue', 'feature_nums', 'local_sampleSize','t')
fprintf('Data is generated\n');

end




 