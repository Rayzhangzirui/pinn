run script_gliomasolve_setup.m
load ./data/datmri_dim2_n40000_st1_linear.mat
load ./data/sol2d_D0.13_rho0.025.mat
% modeldir = '../pinn/models/mri2dscale2/mri2d_stnpos/'
modeldir = '../pinn/models/mri2dscale2/cont/'
load(fullfile(modeldir,'predxdat.mat'));
threshold = false
%%
xqs = double(x*scale(end)) + ix(1:DIM);

minu = -1; % minimum u value to plot
sigma = 0.1;

upredt = upred(:,16);
idx = upredt>minu;
maxu = max(upredt(:));
n = size(upredt,1); 
udat = uq(1:n,1);
udatnoise = udat + randn(n,1)*sigma;
if threshold
    udatnoise(udatnoise<0)=0;
end
err  = abs(upredt - udat); % need to load uq

Dslice = D(:,:,zslice);
phislice = phi(:,:,zslice);

%%
[ax1,himg,ax2,hsc] = plot_mri_over(Dslice,xqs(:,2),xqs(:,1),udatnoise);
title(ax1,'udat with noise over D');
% ax2.CLim = [0 max(uq)] + [0, 2*sigma]
export_fig(fullfile(modeldir,'fig_udat_noise.jpg'),'-m3');

%%

[ax1,himg,ax2,hsc] = plot_mri_over(Dslice,xqs(:,2),xqs(:,1),err);
title(ax1,'err over D');
export_fig(fullfile(modeldir,'fig_err.jpg'),'-m3');

%%
[ax1,himg,ax2,hsc] = plot_mri_over(Dslice,xqs(:,2),xqs(:,1),upredt);
title(ax1,'pred u over phi');
% ax2.CLim = [0 max(uq)] + [0, 2*sigma]
export_fig(fullfile(modeldir,'fig_upred.jpg'),'-m3');

%%
[ax1,himg,ax2,hsc] = plot_mri_over(Dslice,xqs(:,2),xqs(:,1),udat);
title(ax1,'u dat over D');
% ax2.CLim = [0 max(uq)] + [0, 2*sigma]
export_fig(fullfile(modeldir,'fig_udat.jpg'),'-m3');

%%

Finterp = scatteredInterpolant(xqs(:,1),xqs(:,2),upredt,'natural','none');
upredgd = Finterp(gx(:),gy(:));
upredgd = reshape(upredgd,size(gx));

%%
close all

% plot D
ax1 = axes;
colormap(ax1,gray(20));
dimg = imagesc(Dslice);
cb1 = colorbar(ax1,'Location','westoutside');

ax2 = axes;
uimg = imagesc(upredgd);
uimg.AlphaData = double(upredgd>minu); % threshold for transparency
cmp = colormap(ax2,'parula');
caxis(ax2,[minu maxu]); % caxis is clim in newer version of matlab
set(ax2,'color','none','visible','on')
cb2 = colorbar(ax2,'Location','eastoutside')

hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});


