% forward solve using results from pde
%% solver parameters
clear
startup
load model.mat
modeldir = '/Users/Ray/project/glioma/pinn/models/mri2dfwd/mri2d_fwd'

m = PostData(modeldir);
m.unscale(2, g.T, g.L, g.x0);

%%
g.readmri(DIR_MRI)
g.solve
%%
g.plotu

%%
g.scale(0.1, 0.02, double(scale(2)))

%%
[ts, x, upred, ~] = g.loadresults(fullfile(modeldir,'predxdat.mat'));
g.xq = x;

%%
g.uqe = g.interpf(g.uend,x(:,1),x(:,2),'linear')
err = g.uqe - upred;
%%
g.visualover('df',x(:,2),x(:,1),6,err,'MarkerFaceAlpha',0.2)

%% use infered param to solve fdm

g2 = GliomaSolver(DIM, 0.1*1.3442, 0.02*1.2695, x0, tend, zslice)
g2.readmri(DIR_MRI)
g2.solve
%%
g2.uqe = g2.interpf(g2.uend,x(:,1),x(:,2),'linear')
%%
err2 = g2.uqe - upred;

g2.visualover('df',x(:,2),x(:,1),6,abs(err2),'MarkerFaceAlpha',0.2)
%% error between fdm

fdmerr = g2.uqe - g.uqe;

g2.visualover('df',x(:,2),x(:,1),6,abs(fdmerr),'MarkerFaceAlpha',0.2)

