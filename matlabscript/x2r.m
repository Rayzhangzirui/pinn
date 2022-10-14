function r = x2r(varargin)
% transform X to rad

if nargin == 1
    X = varargin{1};
    r = sqrt(sum(X.^2,2));
elseif nargin == 2
    x = varargin{1};
    y = varargin{2};
    r = sqrt(x.^2+y.^2);
end

end