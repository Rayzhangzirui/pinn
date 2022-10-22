function [sol,Rarray] = polar_udat(Tarray,icfun,xdim)
% used to generate data for polar test case
T = 300;
d = 0.13;
rho = 0.025;
L = 50;
EPSILON = 0.1;

D = d*T/L^2;
RHO = rho*T;

ddm = true;
bd = 1.1;

Rarray = linspace(0,bd,101);

icfun = @(x) 0.1*exp(-500*(x).^2);
sol = pde_exact(ddm, EPSILON, xdim-1, D, RHO, Rarray,Tarray, icfun);
end

