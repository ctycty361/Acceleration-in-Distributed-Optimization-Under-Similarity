function e = evaluate(x)
global x_opt I
e = norm(x - x_opt * ones(1, I))^2/I;
% e = norm(x - x_opt * ones(1, I))^2/I/norm(x_opt);