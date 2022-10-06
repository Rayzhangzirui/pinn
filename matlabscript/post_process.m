% visualize 1d pinn simulation
datadir = '/Users/zziruiadmin/projects/glioma/data/model_1d/';

files= sdir([datadir 'data*.dat']);
dim = 1;
tvar = 1;
xvar = 2;
uvar = 3;
resvar = 4;
nvar = 4;
ndat = 10000;
nstep = length(files);
data_interval = 100;

bd = 0.7; % boundary of domain, not boundary of computation box using DDM
dx=0.01;
dt=0.01;
trange = [0,1];
xrange = [-bd,bd];
zrange = [-1,1];

tarray = [0:dt:1];
xarray= [-bd:dx:bd];
[xg,tg] = meshgrid(xarray,tarray);



%% read data into 3d matrix
dat = zeros([ndat nvar length(files)]);

for i = 1:nstep
    tmp = fscanf(fopen(files{i}),'%f',[nvar ndat]);
    dat(:,:,i) = tmp';
end
dat(:,resvar,:) = abs(dat(:,resvar,:));
%% surf plot, need interpolation to grid,  color by abs(residual)
i=1;
ug = griddata(dat(:,xvar,i),dat(:,tvar,i),dat(:,uvar,i),xg,tg); 
objs = surf(xg,tg,ug)
objs.EdgeColor='none';
colorbar
xlabel('x')
ylabel('t')
zlabel('u')

% zlim(zrange)
for i = 1:length(files)
    colordat = dat(:,resvar,i);
    ug = griddata(dat(:,xvar,i),dat(:,tvar,i),dat(:,uvar,i),xg,tg); % u on grid
    resg = griddata(dat(:,xvar,i),dat(:,tvar,i),colordat,xg,tg); % residual on grid
    set(objs,'ZData',ug,'CData',resg);
    title(sprintf('step=%d, residual mse=%e',i*data_interval, mean(colordat.^2)))
    pause(0.2)
end

%% scatter plot, color by abs(residual)
i=1

objsc = scatter3(dat(:,xvar,i),dat(:,tvar,i),dat(:,uvar,i),12,dat(:,resvar,i),'filled');

xlim(xrange)
ylim(trange)
colorbar
xlabel('x')
ylabel('t')
zlabel('u')
% zlim(zrange)
for i = 1:length(files)
    colordat = dat(:,resvar,i);
    set(objsc,'Xdata',dat(:,xvar,i),'YData',dat(:,tvar,i),'ZData',dat(:,uvar,i),'CData',colordat);
    title(sprintf('step=%d, residual mse=%e',i*data_interval, mean(colordat.^2)))
    pause(0.2)
end


%% exact solution
T = 300;
tfinal = 2
D = 0.13e-4;
rho = 0.025;
epsilon = 0.1;
ddm = true;
rarray = linspace(0,0.7,100);
tarray = linspace(0,tfinal,200);
[rg,tg] = meshgrid(rarray,tarray);

solg = pde_exact(ddm, epsilon, dim-1, T,D,rho, rarray,tarray);
%  interpolate grid exact solution to get solution on traning points, use symmetry
solpt = griddata(tg(:),rg(:),solg(:),dat(:,tvar,1), abs(abs(dat(:,xvar,1))));

%%  slice plot
% interpolate data at final iteration, look at error at specific time
err = solpt - dat(:,uvar,end);
scatter3(dat(:,xvar,1),dat(:,tvar,1),dat(:,uvar,end), 12, err, 'filled')
title('err = (exact - pred) at training pts')
colorbar
%%
predictionfile= sdir([datadir 'prediction.dat']);
tmp = fscanf(fopen(predictionfile),'%f',[3 Inf]);
pred  = tmp';
%%
Np = 1001;
ptg = reshape(pred(:,1),Np,Np);
pxg = reshape(pred(:,2),Np,Np);
pug = reshape(pred(:,3),Np,Np);
%%
surf(pxg,ptg,pug,'EdgeColor','none')
