function [x] = sampleDenseBall(N,dim,varargin)
% 

if length(varargin)==1 && isstruct(varargin{1})
    p = varargin{1};
else
    p.radius = 1;
    p.isuniformx = false;
    p.x0 = zeros(1,dim);
    p.issphere = false;
    p = parseargs(p, varargin{:});
end

if ~p.issphere 
    if p.isuniformx
        r = rand(N,1).^(1/dim)*p.radius; %
    else
        r = rand(N,1)*p.radius; % dense at center
    end
else
    r = ones(N,1)*p.radius;
end

% uniform points on sphere
xn = randn(N,dim);
x = xn./vecnorm(xn,2,2);

x = x.*r + reshape(p.x0, 1, dim);
end

