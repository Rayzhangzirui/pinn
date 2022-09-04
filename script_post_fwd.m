run startup.m

% savedir = '/Users/zziruiadmin/projects/glioma/meeting/pinn20220817'

dirprefix = 'models/growth_fwd/';
dirs = {'growth2d_fwd/'};
tags = {'fwd'};

for i = 1:length(dirs)
    p = fullfile(projdir,dirprefix,dirs{i});
    m(i) = Model(p,'tag',tags{i},'isradial',true);
end

%%

% scatter(m(1).predxr.r, m(1).predxr.u, 6, m(1).predxr.t)



scatter3(m(1).predxr.x, m(1).predxr.y, m(1).predxr.u, 6, m(1).predxr.t)