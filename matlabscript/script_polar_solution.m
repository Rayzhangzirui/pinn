% script to solve pde in polar coordinate,
% check sure scaling is correct

xdim = 2;

dw = 0.13;
rho = 0.025;
bd = 50;
EPSILON = 3;
tend = 300;
dt = 60;
dx = 1;
ddm = false;
icfun = @(x) 0.1*exp(-0.1*(x).^2);
[sol,xgrid,tgrid] = pdepolar(ddm, EPSILON, bd, xdim-1, dw, rho, dx, tend, dt, icfun);

ddm = true
[solddm,xgridddm,~] = pdepolar(ddm, EPSILON, bd, xdim-1, dw, rho, dx, tend, dt, icfun);

plot(xgrid, sol)

err = sol(:,1:length(xgrid))-solddm(:,1:length(xgrid));
fprintf('ddm max err %g',max(abs(err(:))));
%% scaled PDE
dwc = 0.1;
rhoc = 0.02;
L = bd;
[V, T, DW, RHO, rDWe, rRHOe] = gscaling(dw, rho, dwc, rhoc, tend, L);
icfuns = @(x) 0.1*exp(-0.1*(x*L).^2);

ddm = true
[sols,xgrids,tgrids] = pdepolar(ddm, EPSILON/L, bd/L, xdim-1, rDWe*DW, rRHOe*RHO, dx/L, tend/tend, dt/tend, icfuns);

plot(xgrids*L, sols)
fprintf('scaling max err %g',max(abs(sols(:)-solddm(:))));

