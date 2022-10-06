%% MRI data
startup
%% solver parameters
DIM = 2;
zslice = 99; % slice for visualization

tend = 150; %day
Dw = 0.13; % mm^2/day
rho= 0.025; %0.025/day
x0 = [164 116];

g = GliomaSolver(DIM, Dw, rho, x0, tend, zslice)
%%
g.readmri(DIR_MRI)
g.solve
g.scale(0.1, 0.02, g.rmax)

%%
g.plotuend

%%
% traindatfile = g.ReadyDat(20000,'method','spline','tag','nzinterp_add','noiseon','uend','type','add')
% traindatfile = g.ReadyDat(20000,'method','spline','tag','nzinterp_add_cor','noiseon','uend','type','add','cor',3)
% traindatfile = g.ReadyDat(20000,'method','spline','tag','nzinterp_mult','noiseon','uend','type','mult','std',1)
% traindatfile = g.ReadyDat(20000,'method','spline','tag','interpnz_add','noiseon','uqe','type','add')
traindatfile = g.ReadyDat(20000,'method','spline','tag','none','noiseon','uqe','type','none')
load(traindatfile)
%%
g.scatter('df', xq, 6, udat-uqe ,'filled')
colorbar
