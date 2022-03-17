function out_x = chebyshev3(W, x, K)
global  I

 
x = x';
L = eye(I)-W;
eigL = sort(eig(L));

% rho = norm( W - 1/I * ones(I,1) * ones(1,I),2);
% chi = 1/(1-rho);

chi = max(eigL)/eigL(2);

T = K;
% T1 = floor(sqrt(chi));

c1 = (sqrt(chi)-1)/(sqrt(chi)+1);
c2 = (chi+1)/(chi-1);
c3 = 2*chi/((1+chi)*max(eigL));
theta = max(eigL)*(1+c1^(2*T))/(1+c1^T)^2;

x00 = x;
a0 = 1;
a1 = c2;
x0 = x;
x1 = c2*(eye(I)-c3*L)*x;
for i = 1:T-1
    a2 = 2*c2*a1-a0;
    x2 = 2*c2*(eye(I)-c3*L)*x1-x0;

    x0 = x1;
    x1 = x2;
    a0 = a1;
    a1 = a2;
end

tmp_out_x = x00-x1/a1;
out_x = x00 - theta*tmp_out_x;
out_x = (out_x)';
end