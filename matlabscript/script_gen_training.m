% generate data to study variance of correlated noise
%
% load model
parentdir = '/Users/Ray/project/glioma/pinn/models/noise_uend_std';
stds = [0.1 0.05 0.01 0]

for i = 1:length(stds)
    childdir = fullfile(parentdir,sprintf('std%g',stds(i)));

    f{i} = g.ReadyDat(5000,'method','spline','datdir',childdir,'tag','std','noiseon','uend','type','add','std',stds(i));
end

%%
childdir = './'
tmp = g.ReadyDat(5000,'method','spline','datdir',childdir,'tag','std','noiseon','uend','type','add','std',1,'cor',1);
load(tmp)
%%
g.scatter('df',xq, 8, udat,'filled')