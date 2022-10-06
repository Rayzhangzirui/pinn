run script_gliomasolve_setup.m

datafname = sprintf('data/sol%dd_D%g_rho%g_x%d_y%d_z%d_tf%d.mat',DIM,Dw,rho,ix(1),ix(2),ix(3),tfinal);
yessave = 'y';
if isfile(datafname)
    fprintf('%s already exist\n',datafname);
    yessave=input('Do you want to continue, y/n [y]:','s');
    if yessave~='y'
        return
    end
end

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

r2 = h.^2*((gx-ix(1)).^2+(gy-ix(2)).^2+(gz-ix(3)).^2); % distance squared
u0 = 0.1*exp(-0.1*r2); 

% "The distribution is chosen such that the initial tumor has an
% approximate radius of 5 mm."
% only 1 pixel 0.1, exp(-10 r^2) is almost zero for r = 5
% imagesc(u0);


dt0 = h^2/(8*max(D(:))); % CFL condition, timestep


% Euler's method in time
u = u0;
t = 0;
tall = [t];
uall = {u0};

DxDphi = Dx(D.*phi);
DyDphi = Dy(D.*phi);
DzDphi = Dz(D.*phi);
    
while t<tfinal
    
    fprintf('%g\n',t)
    if t+dt0<tfinal
        dt = dt0;
    else
        dt = tfinal-t;
    end
    
% simple implementation    
%     Dxu = Dx(u);
%     Dyu = Dy(u);
%     Dzu = Dz(u);
%     dudt = Dx(D.*phi.*Dxu) + Dy(D.*phi.*Dyu) +Dz(D.*phi.*Dzu) + rho*phi.*u.*(1-u);

%  try harmonic mean
%     DDu = hmnLap(D,u);
%     dudt = DDu + rho*phi.*u.*(1-u);


    
    dudt = DxDphi.*Dx(u) + DyDphi.*Dy(u) + DzDphi.*Dz(u) + D.*phi.* Lap(u) + rho*phi.*u.*(1-u);
    

    u = u + dt*dudt;
    u = threshod(u);
    t = t + dt;
    
    
    tall(end+1) = t;
    uall{end+1} = u;
end

uall = cat(DIM+1,uall{:});

%%
if yessave == 'y'
    save(datafname,'uall','tall','u','phi','-v7.3')
end
