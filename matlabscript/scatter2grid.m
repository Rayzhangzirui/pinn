function uq = scatter2grid(X,u,gx,gy,varargin)
% scattered data to grid
assert(size(X,2)==2);
F = scatteredInterpolant(X(:,1), X(:,2), u, varargin{:});
uq = F(gx, gy);
uq(isnan(uq)) = 0;

end