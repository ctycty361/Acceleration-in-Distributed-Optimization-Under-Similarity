function v = SH_localSONATA_Solver(x_local_init, y_delta, U, V, lambda, aux_x, tau, beta, local_eps, step_size) 
% objective function: fi(x) + lambda|x|^2 + beta/2|x - x_local_init|^2 +
% tau/2|x - aux|^2 + <y_delta, x - x_local_init>.

global d
x1 = x_local_init;
x0 = x1 - ones(d,1);

while(local_eps <= norm(x0-x1))
g0 = SmoothHinge_grad(U, V, x1, lambda, (x_local_init*beta + tau*aux_x)/(beta + tau), beta+tau);
g = g0 + y_delta;
x2 = x1 - step_size*g;

x0  = x1;
x1 = x2;
% local_res = norm(x0-x1)
end
v = x1;
end