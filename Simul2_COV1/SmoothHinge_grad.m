function g = SmoothHinge_grad(U, V, w, lambda, x_aux, tau)
% U: size x d, V: size x 1, x: d x 1;

% total = size(U,1);
% g = 0;
% for i = 1:total
%     func_input = V(i,1)*(U(i,:)*w);
%     unit_grad = V(i,1)*unit_SmoothHinge_grad(func_input)*U(i,:)' + lambda*w + tau*(w - x_aux);
%     g = g + unit_grad;
% end
% g = g/total;

total = size(U,1);
func_inputs = V.*(U*w);
derivatives = unit_SmoothHinge_grad(func_inputs);
weighted_derivatives = V.*derivatives;
non_reg_grad = U'*weighted_derivatives/total;
g = non_reg_grad + lambda*w + tau*(w - x_aux);

end

function derivative = unit_SmoothHinge_grad(x)
size_x = size(x);
data_num = size_x(1);
smoothHinge_para = 1;
derivative = (x < (1-smoothHinge_para)*ones(data_num,1)).*(-1) + (x >= (1-smoothHinge_para)*ones(data_num,1)) .* (x < 1) .* ((x-1)/smoothHinge_para);
end
 

