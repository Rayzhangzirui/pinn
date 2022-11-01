function  s = pdepolar(ddm, EPSILON, BD, m, D, RHO, dx, tend, dt, icfun)
% solve reaction diffusion pde,
% with or without normalization
% du/dt = grad ( D grad u) + RHO u(1-u)
% EPSILON = width of diffused domain
% BD = boundary
% ddm = true: use diffused domain
% m = xdim -1 
% sol(i,j) = u(ti,xj)


bd = BD;
if ddm
    bd = BD + EPSILON*5;
end

nx = floor(bd/dx);
s.xgrid = linspace(0,bd,nx+1);
nt = floor(tend/dt);
s.tgrid = linspace(0,tend,nt+1);
s.phi = phi(s.xgrid);
s.sol = pdepe(m,@pdefunc,icfun,@bcfun,s.xgrid,s.tgrid);




% phase field function for diffused domain method
function v = phi(x)
  v = 1;
  if ddm == true
     v = 0.5 + 0.5*tanh((BD - abs(x))/EPSILON);
  end
end
  
% define pde 
function [c,f,s] = pdefunc(x,t,u,dudx) 
c = 1;
f = D*phi(x)*dudx;
s = RHO*u*(1-u);
end

function [pl,ql,pr,qr] = bcfun(xl,ul,xr,ur,t)
% neumann boundary condition
pl = 0;ql = 1;pr = 0;qr = 1;
end

end

