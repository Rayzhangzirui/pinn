function [x] = sampleDenseBall(N,dim,R,x0)
%

r = rand(N,1)*R; %

% uniform points on sphere
xn = randn(N,dim);
x = xn./vecnorm(xn,2,2);

x = x.*r + reshape(x0,1,dim);
end

