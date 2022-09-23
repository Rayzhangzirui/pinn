run script_gliomasolve_setup.m
datafname = sprintf('data/sol%dd_D%g_rho%g.mat',DIM,Dw,rho);
load(datafname);
%% visualization
tk = length(tall);

if DIM==3
    vz = zslice
    uslice = uall(:,:,vz,tk);
    Dslice = D(:,:,vz);
    phislice = phi(:,:,vz);
    uvol = uall(:,:,:,end);
else
    uslice = uall(:,:,tk);
    Dslice = D;
    phislice = phi;
    uvol = uall(:,:,end);
end
maxu = max(uall(:)); % maximum concentration, used to scale color map

close all
% plot D
ax1 = axes;
colormap(ax1,gray(20));
dimg = imagesc(Dslice);
cb1 = colorbar(ax1,'Location','westoutside');

ax2 = axes;
uimg = imagesc(uslice.*phislice);
uimg.AlphaData = double(uslice>0.01); % threshold for transparency
cmp = colormap(ax2,'parula');
caxis(ax2,[0 maxu]); % caxis is clim in newer version of matlab
set(ax2,'color','none','visible','on')
cb2 = colorbar(ax2,'Location','eastoutside')

hLink = linkprop([ax1,ax2],{'XLim','YLim','Position'});

% choose region to sample
idx = find(uvol.*phi>0.01);

distix = sqrt((gx-ix(1)).^2+ (gy-ix(2)).^2+(gz-ix(3)).^2);
[maxdis,maxidx] = max(distix(idx));
% scatter(ax2,gx(idx(maxidx)),gy(idx(maxidx)),'r');
rmax = maxdis+3;
ls = distix - rmax; % level set function
hold on
if DIM==3
    contour(ax2,gy(:,:,vz),gx(:,:,vz),ls(:,:,vz),[0,0]);
else
    contour(ax2,gy,gx,ls,[0,0]);
end
%% interpolate the data
tfinalonly = true;
N = 40000;
rng(1,'twister');
xs = sampleDenseBall(N,DIM,rmax,ix(1:DIM)); % sample coord, unit

if tfinalonly
    %final time only
    ts = ones(N,1)*tfinal; % sample t, unit
else
    ts = rand(N,1)*tfinal; % sample t, unit
end

method = 'linear'


if DIM==3
    [gx,gy,gz,gt] = ndgrid(1:sz(1),1:sz(2),1:sz(3),tall);
    Pwmq = interpn(gx(:,:,:,1),gy(:,:,:,1),gz(:,:,:,1),Pwm,xs(:,1),xs(:,2),xs(:,3),method);
    Pgmq = interpn(gx(:,:,:,1),gy(:,:,:,1),gz(:,:,:,1),Pgm,xs(:,1),xs(:,2),xs(:,3),method);
    phiq = interpn(gx(:,:,:,1),gy(:,:,:,1),gz(:,:,:,1),phi,xs(:,1),xs(:,2),xs(:,3),method);
    uq = interpn(gx,gy,gz,gt,uall,xs(:,1),xs(:,2),xs(:,3),ts,method);
else
    [gx,gy,gt] = ndgrid(1:sz(1),1:sz(2),tall);
    Pwmq = interpn(gx(:,:,1),gy(:,:,1),Pwm,xs(:,1),xs(:,2),method);
    Pgmq = interpn(gx(:,:,1),gy(:,:,1),Pgm,xs(:,1),xs(:,2),method);
    phiq = interpn(gx(:,:,1),gy(:,:,1),phi,xs(:,1),xs(:,2),method);
    uq = interpn(gx,gy,gt,uall,xs(:,1),xs(:,2),ts,method);
end


%% scatter plot of phi
% scatter(xs(:,2),xs(:,1),6,phiq,'filled')
% set(gca,'YDir','reverse')

%%
figure
scatter(xs(:,2),xs(:,1),6,uq,'filled')
set(gca,'YDir','reverse')
%%
scatter(xs(:,1),xs(:,2),6,uq,'filled')

%% 3d
scatter3(xs(:,1),xs(:,2),xs(:,3),6,phiq,'filled')

%% scatter plot of D*phi
Dq = Pwmq*Dw + Pgmq*Dg;
scatter(xs(:,2),xs(:,1),6,Dq.*phiq,'filled')
set(gca,'YDir','reverse')
%%
scatter(xs(:,1),xs(:,2),6,Dq.*phiq,'filled')

%% 3d
scatter3(xs(:,1),xs(:,2),xs(:,3),6,uq,'filled')

%%
xscenter = xs - ix(1:DIM);
datatable = array2table([ts xscenter phiq Pwmq Pgmq uq]);
fname = sprintf('datmri_dim%d_n%d_st%d_%s.txt',DIM,N,tfinalonly,method);
% writetable(datatable,fname,'WriteVariableNames',false)
