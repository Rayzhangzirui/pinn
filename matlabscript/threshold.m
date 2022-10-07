function Y = threshold(X, lb, ub)
% 
Y = X;
Y(Y<lb) = lb;
Y(Y>ub) = ub;
end