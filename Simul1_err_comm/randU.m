function U = randU(n)
% This function generates random unitary matrices if size n

X = randn(n);
[Q,R] = qr(X);
R = diag(diag(R)./abs(diag(R)));
U = Q*R;