startup

DIM = 2;
zslice = 99; % slice for visualization
tend = 150; %day

Dw = 0.1*1.343; % mm^2/day
rho= 0.02*1.2765; %0.025/day
x0 = [164 116];

g1 = GliomaSolver(DIM, Dw, rho, x0, tend, zslice);
g1.readmri(DIR_MRI)
g1.solve()
g1.sample(10000);

%%

uerr = g2.uend-g1.uend;

%%

g2f = g2.interpf(g2.uend, g1.xq(:,1),g1.xq(:,2));
g1f = g1.interpf(g1.uend, g1.xq(:,1),g1.xq(:,2));
