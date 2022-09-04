% used to generate data for polar test case?
%% parameter for PDE
xdim = 2;
tfinal = 300; % final time, [day]
bound = 50; % domain bound, [mm]

tarray = linspace(0,tfinal,301); % time stamp, [day]
rarray = linspace(0,bound,201);  % r grid, [mm]
ddmwidth = 5; % width of diffused domain, [mm]
bd = bound + ddmwidth * 2;
d = 0.13; % diffusion coefficient
rho = 0.025; % growth rate

ddm = true;

scalemethod = 1;

N = 20000;

%% normalize by final time and bound, so that inputs are [-1 1]
if scalemethod == 1
    T = tfinal;
    L = bound;
    D = d*T/L^2;
    RHO = rho*T;
end 

%% different way of scaling, so that the parameters D and rho are 1
if scalemethod ==2
    T = 1/rho;
    L = sqrt(d/rho);
    D = 1;
    RHO = 1;
end 


%%
BD = bd/L;
EPSILON = ddmwidth/L;
Tarray = tarray/T;
Rarray = rarray/L;

% 0.1*exp(-0.1*x.^2) chosen so that if x is [mm], initial bump has radius about 5
icfun = @(x) 0.1*exp(-0.1*(x*L).^2); 
sol = pde_exact(ddm, EPSILON, BD, xdim-1, D, RHO, Rarray,Tarray, icfun);


%%
% surf(rarray,tarray,sol)
%%
close all
fig = figure;
ax = axes;
hold(ax,'on')

nts = length(tarray)-1;% number of time interval
step = ceil(nts/4);
tidx = [1 step+1 2*step+1 3*step+1 nts+1]; 

for j = 1:length(tidx)
    p(j) = plot(ax,rarray, sol(tidx(j),:),...
        'DisplayName',sprintf('t=%g',tarray(tidx(j))),...
        'LineWidth',2);
    s(j).MarkerFaceColor = p(j).Color;
    s(j).Annotation.LegendInformation.IconDisplayStyle = 'off';
end
ylim([0,1])
xlim([0,bound])
legend('Location','best')

%% sampling, all the time
% the solution first go down/diffuse, then grow. So okay to have sampled data 
% going below inital guess

N = 20000;
Ts = Tarray(end)*rand(N,1); % t' sample
Rs = rand(N,1)*Rarray(end); % r' sample
theta = rand(N,1)*2*pi;

% r ,x, y, t, variable with unit
t = Ts * T;
r = Rs * L; 
x = Rs.*cos(theta)*L;
y = Rs.*sin(theta)*L;
uinterp = interp2(Rarray, Tarray, sol, Rs, Ts,'cubic');
% scatter3(x*L,y*L,uinterp,[],Ts)
scatter(r,uinterp,6,t,'filled')


datatable = array2table([t x y uinterp]);
fname = sprintf('exactu_dim%d_n%d_unscale.txt',xdim,N);
% writetable(datatable,fname,'WriteVariableNames',false)
%%
% figure
% surf(Rarray,Tarray,sol,'EdgeColor','none')
% hold on
% scatter3(Rs,Ts,uinterp,6,'k','filled')

%% Sampling, only final time
N = 20000;
Ts = Tarray(end)*ones(N,1); % t' sample
Rs = rand(N,1)*Rarray(end); % r' sample
theta = rand(N,1)*2*pi;

% r ,x, y, t, variable with unit
t = Ts * T;
r = Rs * L; 
x = Rs.*cos(theta)*L;
y = Rs.*sin(theta)*L;
uinterp = interp2(Rarray, Tarray, sol, Rs, Ts,'cubic');

scatter(x,y,6,uinterp,'filled')

datatable = array2table([t x y uinterp]);
fname = sprintf('exactu_dim%d_n%d_unscale_tfinal.txt',xdim,N);
writetable(datatable,fname,'WriteVariableNames',false)