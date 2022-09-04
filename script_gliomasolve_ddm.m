% tumor growth using diffused domain method
% work both in 3d and 2d
%% MRI data
DIR_JANA_DATA_PROC = '/Users/zziruiadmin/projects/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC)) % need MRIread
DIR_MRI = '/Users/zziruiadmin/projects/glioma/jana/Atlas/anatomy/'
gm = MRIread( [DIR_MRI 'GM.nii']);
wm = MRIread( [DIR_MRI 'WM.nii']);
csf = MRIread( [DIR_MRI 'CSF.nii']);
%%
DIM = 2;
zslice = 99; % slice for visualization
Pwm = wm.vol;
Pgm = gm.vol;
Pcsf = csf.vol;

if DIM == 2
%%% for 2d case, slice 3d MRI, then set zslice = 1, the third dimension is
%%% just 1. Filter with periodic padding make all z-derivative 0. So same as 2D
    Pwm = Pwm(:,:,zslice);
    Pgm = Pgm(:,:,zslice);
    Pcsf = Pcsf(:,:,zslice);
    zslice = 1;
end


%% solver parameters
tfinal = 150; %day
Dw = 0.13; % mm^2/day
Dg = Dw/10;
rho= 0.025; %0.025/day
% x0 = [121 160]; % initial location
x0 = [116 164];

h = 1; % spacial resolution, mm (caption figure 1)
epsilon = 3; % width of diffused domain

%% finite difference operators
flap = cat(3,[0 0 0;0 1 0; 0 0 0],[0 1 0;1 -6 1; 0 1 0],[0 0 0;0 1 0; 0 0 0])/h^2;
fdx = [-1 0 1]/(2*h);
fdy = fdx';
fdz = reshape(fdx,1,1,3);
operator = @(x,f) imfilter(x,f,'circular','same'); % periodic bc

Dx = @(x) operator(x,fdx);
Dy = @(x) operator(x,fdy);
Dz = @(x) operator(x,fdz);
Lap = @(x) operator(x,flap);

threshod = @(x) min(max(x,0),1); % threshold in [0,1]
%% Solve Cahn-Hilliard equation to get diffused domain

% if Pwm + Pgm > Pcsf or phi_threshold, then it's tissue 
% take indicator function of tissue region as initial guess of CH equation
phi_threshold = 0.1; 
phi = double( Pwm+Pgm> max(phi_threshold,Pcsf)); 


numiter = 100; % number of Cahn Hillard steps
dt = h^4/(8*2*epsilon); % dt condition, see Glioma_ComputePFF_CahnHilliard.cpp
tau = 1e-7; %  see CahnHillardOperator, tau = 0.001 (some small number, needed if psi(t=0) = {0,1})

% Euler's method in time
for i = 1:numiter
    fprintf('%g\n',i)
    
    lap_phi = Lap(phi); 
    u = (1/2)*phi.*(1-phi).*(1-2*phi)-epsilon^2*lap_phi;
    M = phi.*(1-phi) + tau;
    dxu = M.*Dx(u);
    dyu = M.*Dy(u);
    dzu = M.*Dz(u);
    dtphi = Dx(dxu)+Dy(dyu)+Dz(dzu);
    phi = phi + dt*dtphi;
    phi = threshod(phi);
end

%% visualize diffused domain
imagesc(phi(:,:,zslice));
%% Solve reaction diffusion 

% generate initial condition, pixel index
sz = [1 1 1];
sz(1:numel(size(Pwm))) = size(Pwm); % even when Pwm is 2d, 3rd dim is 1

% integer grid, horizontal is x, vertical y, only used to get u0
[gx,gy,gz] = meshgrid(1:sz(2),1:sz(1),1:sz(3)); 

ix = [x0 zslice]; % pixel index of initial tumor
r2 = h.^2*((gx-ix(1)).^2+(gy-ix(2)).^2+(gz-ix(3)).^2); % distance squared
u0 = 0.1*exp(-0.1*r2); 

% "The distribution is chosen such that the initial tumor has an
% approximate radius of 5 mm."
% only 1 pixel 0.1, exp(-10 r^2) is almost zero for r = 5
% imagesc(u0);

D = Pwm*Dw + Pgm*Dg; % diffusion coefficients

dt0 = h^2/(4*max(D(:))); % CFL condition, timestep


% Euler's method in time
u = u0;
t = 0;
tall = [t];
uall = {u0};
while t<tfinal
    
    fprintf('%g\n',t)
    if t+dt0<tfinal
        dt = dt0;
    else
        dt = tfinal-t;
    end
    
    
    Dxu = Dx(u);
    Dyu = Dy(u);
    Dzu = Dz(u);
    dudt = Dx(D.*phi.*Dxu) + Dy(D.*phi.*Dyu) +Dz(D.*phi.*Dzu) + rho*phi.*u.*(1-u);
    u = u + dt*dudt;
    u = threshod(u);
    t = t + dt;
    
    
    tall(end+1) = t;
    uall{end+1} = u;
end

uall = cat(DIM+1,uall{:});
%% visualization
vz = 99; % vz is the z-slice for  visualization
if DIM==2
    vz = 1;
end
maxu = max(u(:)); % maximum concentration, used to scale color map

Dslice = D(:,:,vz);
uslice = u(:,:,vz);

close all
% plot D
ax1 = axes;
colormap(ax1,gray(20));
dimg = imagesc(Dslice);
cb1 = colorbar(ax1,'Location','westoutside');

ax2 = axes;
uimg = imagesc(uslice.*phi);
uimg.AlphaData = double(uslice>0.01); % threshold for transparency
cmp = colormap(ax2,'parula');
caxis(ax2,[0 maxu]); % caxis is clim in newer version of matlab
set(ax2,'color','none','visible','on')
cb2 = colorbar(ax2,'Location','eastoutside')

hLink = linkprop([ax1,ax2],{'XLim','YLim','Position'});
%% choose region to sample
idx = find(uslice.*phi>0.01);
distix = sqrt(sum(([gx(:) gy(:) gz(:)] - ix).^2,2));
[maxdis,maxidx] = max(distix(idx));
% scatter(ax2,gx(idx(maxidx)),gy(idx(maxidx)),'r');
rmax = maxdis+3;
viscircles(x0,rmax)

%% interpolate the data
tfinalonly = true;
N = 20000;
rng(1);
xs = sampleDenseBall(N,DIM,rmax,x0); % sample coord, unit
if tfinalonly
    %final time only
    ts = ones(N,1)*tfinal; % sample t, unit
else
    ts = rand(N,1)*tfinal; % sample t, unit
end
    
[gxt,gyt,gt] = meshgrid(1:sz(2),1:sz(1),tall); 
[gx2d,gy2d] = meshgrid(1:sz(2),1:sz(1)); 
Pwmq = interp2(gx2d,gy2d,Pwm,xs(:,1),xs(:,2));
Pgmq = interp2(gx2d,gy2d,Pgm,xs(:,1),xs(:,2));
phiq = interp2(gx2d,gy2d,phi,xs(:,1),xs(:,2));
uq = interp3(gxt,gyt,gt,uall,xs(:,1),xs(:,2),ts);

%%
scatter(xs(:,1),xs(:,2),6,phiq,'filled')
%%
Dq = Pwmq*Dw + Pgmq*Dg;
scatter(xs(:,1),xs(:,2),6,Dq.*phiq,'filled')
%%
scatter3(xs(:,1),xs(:,2),uq,12,ts,'filled')

%%
xscenter = xs - x0; % center x0
datatable = array2table([ts xscenter phiq Pwmq Pgmq uq]);
fname = sprintf('datmri_dim%d_n%d_st%d.txt',DIM,N,tfinalonly);
% writetable(datatable,fname,'WriteVariableNames',false)
