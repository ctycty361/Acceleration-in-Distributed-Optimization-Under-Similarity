function cov_data_generator(data_size, I, K, step_size, lambda, filename)
% data_size: total number of instances chosen
% I: number of agents
% K: the number of iterations of centralized gradient descent steps
% step_size (0.05): the step_size adopted by the gradient descent algorithm


tic
rng(112019,'v4');

%%% Shuffle data %%%

original_data_file = 'cov1_original_data.mat';
if(~exist(original_data_file))
    [cov1_labels, cov1_features] = libsvmread('covtype.libsvm.binary.scale');
    % The dataset covtype.libsvm.binary.scale can be downloaded from
    % https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html 
    save(original_data_file,'cov1_labels','cov1_features');
end
load(original_data_file);

U_stack = cov1_features; 
V_stack = cov1_labels;
[num_instances, feature_nums] = size(cov1_features);

rand_instance_idx = randperm(num_instances);
chosen_idx = rand_instance_idx(1:data_size);
 
U_stack = U_stack(chosen_idx,:);
V_stack = V_stack(chosen_idx,:);
V_stack(V_stack == 2) = -1;

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
gradnorm = zeros(K,1);
 
x_aux = zeros(d,1);
tau = 0;
for k = 1:K
     
    G = zeros(d,I);
    for i = 1:I
    g           =  SmoothHinge_grad(U{i}, V{i}, x, lambda, x_aux, tau);
    G(:,i) = g;
    end
    g = mean(G,2);
    
    x           =  x - step_size * g;
    gradnorm(k) = norm(g);
     
    timegap = 1;
    if mod(k, timegap) == 0
        fprintf('gradient descent: %d-th iteration, the error is %f\n', k, gradnorm(k));
    end
end

% SH_Fopt = SmoothHinge_Fvalue(K);
SH_Xopt = x;

%%%%% plot %%%%%
figure;

subplot(1,1,1);
semilogy(1:K, gradnorm);
xlabel('Iteration')
ylabel('SH-Gradient Norm')
 
t = toc
% name = ['cov1_data_opt_',num2str(data_size), '.mat'];
save(filename, 'lambda','V_stack','U_stack','U', 'V', 'SH_Xopt', 'feature_nums', 'local_sampleSize','t')
fprintf('Data is generated\n');

end