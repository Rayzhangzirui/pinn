function  pygendata(modelpath, varargin)
% run from python
    s = load(modelpath);
    traindatfile = s.g.ReadyDat(varargin{:});
end