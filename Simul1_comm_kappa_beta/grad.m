function g = grad(i, x, tau, aux_x)
global Q b lambda
% g = Q(:,:,i) * x - b(:,i) + lambda * x + tau * (x - aux_x);
g = Q(:,:,i) * x - b(:,i) + lambda * x + tau * (x - aux_x);
end
