function [nzu] = addNoise(u, varargin)
% add noise

    p = inputParser;
    p.KeepUnmatched = true;
    addParameter(p, "mu", 0);
    addParameter(p, "std", 0.1);
    addParameter(p, "cor", 0);
    addParameter(p, "type", 'none');
    parse(p, varargin{:});
    mu = p.Results.mu;
    std = p.Results.std;
    cor = p.Results.cor;
    nztype = p.Results.type;
    grand = randn(size(u))*std + mu;
    
    if strcmpi(nztype,'none')
        fprintf('no noise\n')
        nzu = u;
        return 
    end

    if cor > 0 && length(size(grand))>1
        fprintf('apply filter with cor %d\n',2*cor)
        grand = imgaussfilt3(grand,cor,'FilterDomain','spatial');
    end
    
    fprintf('add noise of type %s, mu %g, std %g \n', nztype,mu, std);
    if strcmpi(nztype,'add')
        nzu = u + grand;
    elseif strcmpi(nztype,'mult')
        nzu = u .* (1 + grand);
    elseif strcmpi(nztype,'logn')
        nzu = u + exp(grand);
    else
        error('unknow type')    
    end

    fprintf('lower bound by 0\n');
    nzu(nzu<0) = 0;
    

end