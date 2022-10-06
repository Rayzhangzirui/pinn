% solver burgers equation
rng(1234)

m = 0;

x = linspace(-1,1,1001);
t = linspace(0,1,1001);
sol = pdepe(m,@bpde,@ic,@bc,x,t);


N = 64^2;
st = rand(N,1);
sx = rand(N,1)*2-1;
uinterp = interp2(x, t, sol, sx, st);
scatter(sx,uinterp,[],st)


T = array2table([st sx uinterp]);
fname = sprintf('burgers_n%d.txt',N);
writetable(T,fname,'WriteVariableNames',false)

function [c,f,s] = bpde(x,t,u,dudx)
    nu = 0.01/pi;
    c = 1;
    f = nu*dudx;
    s = -u*dudx;
end

function u0 = ic(x)

u0 = -sin(pi*x);
end

function [pl,ql,pr,qr] = bc(xl,ul,xr,ur,t)
pl = ul;
ql = 0;
pr = ur;
qr = 0;
end