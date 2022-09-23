load ../pinn/models/mri3d/mri3dst/predxdat.mat
%%
xqs = double(x)*18.841097 + ix;
%%
close all
ax1 = axes;
dimg = imagesc(Dslice);
colormap(ax1,gray(20));
cb1 = colorbar(ax1,'Location','westoutside');
axis(ax1,'equal')
xlim(ax1,[0 size(Dslice,2)])
%%
ax2 = axes;
s = scatter(ax2,xqs(:,2),xqs(:,1),3,upred(:,end),'filled')
cmp = colormap(ax2,'parula');
set(ax2,'YDir','reverse')
% caxis(ax2,[0 1]); % caxis is clim in newer version of matlab
set(ax2,'color','none','visible','off')
cb2 = colorbar(ax2,'Location','eastoutside')
% 
hLink = linkprop([ax1,ax2],{'XLim','YLim','Position'});
%%

Finterp = scatteredInterpolant(xqs(:,1),xqs(:,2),upred(:,10),'natural','none');
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
uimg = imagesc(upredgd.*phislice);
uimg.AlphaData = double(upredgd>0.01); % threshold for transparency
cmp = colormap(ax2,'parula');
caxis(ax2,[0 maxu]); % caxis is clim in newer version of matlab
set(ax2,'color','none','visible','on')
cb2 = colorbar(ax2,'Location','eastoutside')

hLink = linkprop([ax1,ax2],{'XLim','YLim','Position'});


