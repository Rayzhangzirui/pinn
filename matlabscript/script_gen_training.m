% generate data to study variance of correlated noise
% load model
parentdir = '/Users/Ray/project/glioma/pinn/models/noise_uend_cor';
cors = [0 1 2 3]

for i = 1:length(cors)
    childdir = fullfile(parentdir,sprintf('cor%g',cors(i)));

    f{i} = g.ReadyDat(5000,'method','spline','datdir',childdir,'tag','cor','noiseon','uend','type','add','std',0.01,'cor',cors(i));
end

%%
load(f{2})
g.scatter('df',xq, 8, udat,'filled')