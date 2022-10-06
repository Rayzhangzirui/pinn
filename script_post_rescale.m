run startup.m
fudat = projpath('models/growth2d_rescale/exactu_dim2_n20000.txt')
savedir = '/Users/zziruiadmin/projects/glioma/meeting/pinn20220825'

dirprefix = 'models/growth2d_rescale/';
dirs = {'growth2d_inv_rescale/','growth2d_inv_rescale_loc'};
tags = {'inv','inv-loc'};

for i = 1:length(dirs)
    p = fullfile(projdir,dirprefix,dirs{i})
    m(i) = Model(p,'tag',tags{i},'udat',fudat);
end

%% inverse problem parameter

close all
fig = figure;
ax = axes;
hold(ax,'on')

eRHO = 1;
eD = 1; 

a = 2

% relerr = @(x,b) abs(x-b)./b;

relerr = @(x,b) (x-b)./b;
abserr = @(x,b) abs(x-b)

errD = relerr(m(a).log.rD,eD);
errRHO = relerr(m(a).log.rRHO,eD);

% plot(ax, m(a).log.it, errD ,'DisplayName','PINN D')
% plot(ax, m(a).log.it, errRHO,'DisplayName','PINN \rho')




title(['rel err ' tags{a} sprintf(' D = %.1e, RHO = %.2e',errD(end),errRHO(end))])

xlabel('step')
legend('Location','best')
grid on
%%
export_fig(fullfile(savedir,'fig_inv_loc_gpinn_param.pdf'))
%%

close all
fig = figure;
ax = axes;
hold(ax,'on')

plot(ax, m(a).log.it, abserr(m(a).log.x0,0) ,'DisplayName','x0')
plot(ax, m(a).log.it, abserr(m(a).log.y0,0) ,'DisplayName','y0')

title('infer x0 y0')

xlabel('step')
legend('Location','best')
grid on
%%
export_fig(fullfile(savedir,'fig_inv_loc_gpinn_ic.pdf'))

%%
i = 2;
m(i).udat.r = sqrt(sum(m(i).udat{:,2:3}.^2,2));
scatter(m(i).udat.r, m(i).predxr.Var1, 6, m(i).udat.Var1,'filled')
% scatter3(m(i).udat.Var2, m(i).udat.Var3, m(i).predxr.Var1, 6, m(i).udat.Var1,'filled')
%%
scatter(m(i).predt.x0, m(i).predt.x1, 6, m(i).predt{:,6}, 'filled')


