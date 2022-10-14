function [x] = sampleDenseBall(N,dim,R,x0,isuniform)
% 

if nargin < 5 
    isuniform = true;
end

if nargin < 4
    x0 = zeros(1,dim);
end



if isuniform
    r = rand(N,1).^(1/dim)*R; %
else
    r = rand(N,1)*R; % dense at center
end

% uniform points on sphere
xn = randn(N,dim);
x = xn./vecnorm(xn,2,2);

x = x.*r + reshape(x0,1,dim);
end

