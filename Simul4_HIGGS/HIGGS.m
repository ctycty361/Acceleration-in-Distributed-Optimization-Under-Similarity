function [] = HIGGS(N, HIGGS_features, HIGGS_labels)

rand_instance_idx = randperm(N);
chosen_idx = rand_instance_idx(1:N);
HIGGS_features1 = HIGGS_features(chosen_idx,:);
HIGGS_labels1 = HIGGS_labels(chosen_idx,:);

save(['HIGGS_',num2str(N),'.mat'], 'HIGGS_features1', 'HIGGS_labels1');

end
