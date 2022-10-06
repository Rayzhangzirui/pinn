function sol = pde_exact(ddm, EPSILON, BD, m, D, RHO, xgrid,tgrid, icfun)
% solve reaction diffusion pde,
% Normalized Time and Length
% sol(i,j) = u(ti,xj)

sol = pdepe(m,@pdefunc,icfun,@bcfun,xgrid,tgrid);


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

% 
% function u0=icfun(x) 
%     u0=0.1*exp(-100*(x).^2);
% end


function [pl,ql,pr,qr] = bcfun(xl,ul,xr,ur,t)
% neumann boundary condition
pl = 0;ql = 1;pr = 0;qr = 1;
end

end

