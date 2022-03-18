function mnist_data_generator(data_size, I, K, step_size, lambda, filename, goal, data_eps)
% data_size: total number of instances chosen
% I: number of agents
% K: the number of iterations of centralized gradient descent steps
% step_size (0.05): the step_size adopted by the gradient descent algorithm
 
tic
rng(112019,'v4');



%%% Shuffle data %%%
data_name = ['MNIST_',num2str(goal),'.mat'];
if(~data_name)
    imageFile = 'train-images.idx3-ubyte'; 
    labelFile = 'train-labels.idx1-ubyte';
    features = loadMNISTImages(imageFile); 
    % The file "train-images.idx3-ubyte" and "train-labels.idx1-ubyte" can
    % be downloaded from http://yann.lecun.com/exdb/mnist/.
    labels = loadMNISTLabels(labelFile);
    
    features = features';

    no4_idx = find(labels ~= goal);
    size_data = size(labels,1);
    MNIST_labels = ones(size_data, 1);
    MNIST_labels(no4_idx) =-1;
    MNIST_features = features;
    save(['MNIST_',num2str(goal),'.mat'],'MNIST_labels','MNIST_features');
else
    load(['MNIST_',num2str(goal),'.mat']);
end
 
U_stack = MNIST_features;
V_stack = MNIST_labels;
 
[num_instances, feature_nums] = size(MNIST_features);
 

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
 
SH_Fdesc = {};
SH_Fvalue = {};

x_aux = zeros(d,1);
tau = 0;
% for k = 1:K
iter = 0;
grad_offset = 1;
while(grad_offset > data_eps)
%     grad_offset
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
% name = ['cov1_data_opt_',num2str(data_size), '.mat'];
save(filename, 'lambda','V_stack','U_stack','U', 'V', 'SH_Fopt', 'SH_Xopt', 'SH_Fvalue', 'feature_nums', 'local_sampleSize','t')
fprintf('Data is generated\n');

end




 