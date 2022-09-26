function [ax1,himg,ax2,hsc] = plot_mri_over(Dslice,x,y,u)
% overlay plots on mri slices

close all
ax1 = axes;
himg = imagesc(Dslice);
colormap(ax1,gray(20));
cb1 = colorbar(ax1,'Location','westoutside');
axis(ax1,'equal')
xlim(ax1,[0 size(Dslice,2)])

ax2 = axes;
hsc = scatter(ax2,x,y,3,u,'filled');
hsc.MarkerFaceAlpha = 0.2;
cmp = colormap(ax2,'parula');
set(ax2,'YDir','reverse')
% caxis(ax2,[minu maxu]); % caxis is clim in newer version of matlab
set(ax2,'color','none','visible','off')
cb2 = colorbar(ax2,'Location','eastoutside')
% 
hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});

end

