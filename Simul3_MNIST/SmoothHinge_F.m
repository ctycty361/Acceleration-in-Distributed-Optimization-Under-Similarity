function fvalue = SmoothHinge_F(U, V, w, lambda, x_aux, tau)
% U: size x d, V: size x 1, x: d x 1;
func_inputs = V.*(U*w);
fvalue = mean(unit_SmoothHinge_F(func_inputs)) + lambda/2*norm(w)^2 + tau/2 * norm(w - x_aux)^2;
end

function unit_value = unit_SmoothHinge_F(x)
% x must be num x 1
smoothHinge_para = 1;
unit_value = ( x < 1-smoothHinge_para).*(1 - x - smoothHinge_para/2) + ( (x >= 1-smoothHinge_para).* (x < 1) ).* (1/(2*smoothHinge_para)*(1-x).^2);
end