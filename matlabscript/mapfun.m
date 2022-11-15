function [varargout] = mapfun(f,varargin)
% apply transformation to multiple arguments
% out_k = f(in_k)
    assert(nargout+1==nargin);
    
    if length(varargin) == 1 && isstruct(varargin{1})
        % apply to each field
        s = varargin{1};
        fnames = fieldnames(s);
        for i = 1:length(fnames)
            s.(fnames{i}) = f(s.(fnames{i}));
        end
        varargout{1} = s;
        return
    end

    for i = 1:nargout
        varargout{i} = f(varargin{i});
    end

end