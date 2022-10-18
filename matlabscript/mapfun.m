function [varargout] = mapfun(f,varargin)
% apply transformation to multiple arguments
    assert(nargout+1==nargin);
    for i = 1:nargout
        varargout{i} = f(varargin{i});
    end

end