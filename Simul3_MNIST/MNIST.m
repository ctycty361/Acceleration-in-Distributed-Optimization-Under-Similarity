imageFile = 'train-images.idx3-ubyte';
labelFile = 'train-labels.idx1-ubyte';
features = loadMNISTImages(imageFile);
labels = loadMNISTLabels(labelFile);

features = features';

goal = 4;
no4_idx = find(labels ~= goal);

size_data = size(labels,1);
MNIST_labels = ones(size_data, 1);
MNIST_labels(no4_idx) =-1;
MNIST_features = features;
save(['MNIST_',num2str(goal),'.mat'],'MNIST_labels','MNIST_features');

