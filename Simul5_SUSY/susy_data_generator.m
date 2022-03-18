function susy_data_generator(data_size, I, K, step_size, lambda, filename,  data_eps)

tic
rng(112019,'v4');

original_data_file = 'SUSY_data.mat';
if(~exist(original_data_file))
    [SUSY_labels, SUSY_features] = libsvmread('SUSY');
    % The dataset SUSY can be found at https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    save(original_data_file,'SUSY_labels','SUSY_features');
else
    load('SUSY_data.mat');
end
U_stack = SUSY_features;
V_stack = SUSY_labels;
[num_instances, feature_nums] = size(SUSY_features);

rand_instance_idx = randperm(num_instances);
chosen_idx = rand_instance_idx(1:data_size);

U_stack = U_stack(chosen_idx,:);
V_stack = V_stack(chosen_idx,:);
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

Logis_Fdesc = {};
Logis_Fvalue = {};

x_aux = zeros(d,1);
tau = 0;
 
iter = 0;
grad_offset = 1;
while(grad_offset > data_eps)
    %     grad_offset
    iter = iter + 1;
    
    G = zeros(d,I);
    for j = 1:I
        G(:,j) = Logis_grad(U{j}, V{j}, x, lambda,  x_aux, tau);
    end
    g = mean(G,2);
    
    x           =  x - step_size * g;
    gradnorm{iter} = norm(g);
    grad_offset = gradnorm{iter};
    
    
    F = zeros(d,I);
    for j = 1:I
        F(:,j) = Logis_F(U{j},V{j}, x, lambda, x_aux, tau);
    end
    
    Logis_Fvalue{iter} = mean(F,2);
    
    if iter > 1
        
        Logis_Fdesc{iter-1} = abs(Logis_Fvalue{iter-1} - Logis_Fvalue{iter});
    end
    timegap = 1;
    if mod(iter, timegap) == 0
        fprintf('gradient descent: %d-th iteration, the error is %f\n', iter, gradnorm{iter});
    end
end

Logis_Fopt = Logis_Fvalue{end};
Logis_Xopt = x;

%%%%% plot %%%%%
figure;

subplot(1,3,1);
semilogy(1:iter, cell2mat(gradnorm));
xlabel('Iteration')
ylabel('Logis-Gradient Norm')

subplot(1,3,2);
semilogy(1:iter, cell2mat(Logis_Fvalue));
xlabel('Iteration')
ylabel('Logis-F value')

subplot(1,3,3);
semilogy(1:iter-1, cell2mat(Logis_Fdesc));
xlabel('Iteration')
ylabel('descent per step of Logis_F value')
t = toc

save(filename, 'lambda','V_stack','U_stack','U', 'V', 'Logis_Fopt', 'Logis_Xopt', 'Logis_Fvalue', 'feature_nums', 'local_sampleSize','t')
fprintf('Data is generated\n');


end