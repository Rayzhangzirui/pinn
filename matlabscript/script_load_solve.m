% forward solve using results from pde
%% solver parameters
clear
startup
modeldir = '~/project/glioma/pinn/models/noise_uend_std/std0.01cont'

load(fullfile(modeldir,'predxdat.mat'))

DIM = 2;
zslice = 99; % slice for visualization
tend = 150; %day
x0 = [164 116];

%% use infered param to solve fdm

g2 = GliomaSolver(DIM, 0.1*1.3442, 0.02*1.2695, x0, tend, zslice)
g2.readmri(DIR_MRI)
g2.solve
g2.scale(0.1, 0.02, double(scale(2)))
%%
g2.plotu
title('fdm with inferred parameters')
caxis([0,0.33])
export_fig(fullfile(modeldir,'fig_fdm_infer_param.jpg'),'-m3');
%%
[ts, x, upred, ~] = g2.loadresults(fullfile(modeldir,'predxdat.mat'));
g2.uqe = g2.interpf(g2.uend,x(:,1),x(:,2),'spline')
%%

close all
idx = upred>0.01;
[h1,ax1,h2,ax2] = g2.visualover('df',x(idx,2),x(idx,1),6, upred(idx),'MarkerFaceAlpha',0.1)

axes(ax2);
caxis([0,0.33])

title(ax2,'pinn u_end')
export_fig(fullfile(modeldir,'fig_pinn_infer_param.jpg'),'-m3');

%%
g2.visualover('df',x(:,2),x(:,1),6, g2.uqe,'MarkerFaceAlpha',0.1)
title('interp fde u_end')
caxis([0,0.33])
    
%%
err2 = (g2.uqe - upred)

g2.visualover('df',x(:,2),x(:,1),6,abs(err2),'MarkerFaceAlpha',0.2)



%% u from original data
Dw = 0.1*1.3; % mm^2/day
rho= 0.02*1.25; %0.025/day
x0 = [164 116];

go = GliomaSolver(DIM, Dw, rho, x0, tend, zslice);
go.readmri(DIR_MRI)
go.solve()
%%
go.plotu
title('fdm with original parameters')
caxis([0,0.33])
export_fig(fullfile(modeldir,'fig_fdm_origin_param.jpg'),'-m3');

%%
load '../pinn/models/mri2dscale2/mri2d_stnpos/predxdat.mat'
[ts, x, upred, ~] = g2.loadresults('../pinn/models/mri2dscale2/mri2d_stnpos/predxdat.mat');