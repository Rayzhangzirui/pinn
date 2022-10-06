run startup.m
fudat = projpath('models/growth2d_xgrad/exactu_dim2_n20000.txt')
savedir = '/Users/zziruiadmin/projects/glioma/meeting/pinn20220817'

dirprefix = 'models/growth2d_xgrad/';
dirs = {'growth2d_fwd','growth2d_inv','growth2d_fwd_xgrad','growth2d_inv_xgrad'};
tags = {'fwd','inv','fwd-xgrad','inv-xgrad'};

for i = 1:length(dirs)
    p = fullfile(projdir,dirprefix,dirs{i})
    m(i) = Model(p,'tag',tags{i},'udat',fudat);
end
%% foward problem, loss

close all
fig = figure;
ax = axes;
hold(ax,'on')

plot(ax, m(1).log.it, log10(m(1).log.res),'DisplayName','PINN res')

plot(ax, m(3).log.it, log10(m(3).log.res),'DisplayName','gPINN res')
plot(ax, m(3).log.it, log10(m(3).log.dres * 1e-2),'DisplayName','gPINN 0.01gres')
plot(ax, m(3).log.it, log10(m(3).log.total),'DisplayName','gPINN res + 0.01\nablares')

fprintf('PINN mse %e\n',mean((m(1).predxr{:,1} - m(1).udat{:,end}).^2))
fprintf('gPINN mse %e\n',mean((m(3).predxr{:,1} - m(3).udat{:,end}).^2))


xlabel('step')
ylabel('$$\log_{10}$$(loss)')
legend('Location','best')
grid on
%%
export_fig(fullfile(savedir,'fig_fwd_gpinn_loss.pdf'))
%% inverse problem parameter

close all
fig = figure;
ax = axes;
hold(ax,'on')
a = 2;
b = 4;

eRHO = 7.5;
eD = 0.0156; 

% plot(ax, m(a).log.it, log10(m(a).log.res),'DisplayName','PINN res')
% plot(ax, m(a).log.it, log10(m(a).log.data),'DisplayName','PINN data')
% plot(ax, m(a).log.it, log10(m(a).log.total),'DisplayName','PINN total')
% 
% plot(ax, m(b).log.it, log10(m(b).log.res),'DisplayName','gPINN res')
% plot(ax, m(b).log.it, log10(m(b).log.data),'DisplayName','gPINN data')
% plot(ax, m(b).log.it, log10(m(b).log.total),'DisplayName','gPINN total')

relerr = @(x,b) abs(x-b)./b;

v1 = 'history.var1'

plot(ax, m(a).history.Var1, relerr(m(a).history.Var6,eD) ,'DisplayName','PINN D')
plot(ax, m(a).history.Var1, relerr(m(a).history.Var7,eRHO) ,'DisplayName','PINN \rho')

plot(ax, m(b).history.Var1, relerr(m(b).history.Var6,eD),'--' ,'DisplayName','gPINN D')
plot(ax, m(b).history.Var1, relerr(m(b).history.Var7,eRHO),'--' ,'DisplayName','gPINN \rho')
title('relative error of parameters')

xlabel('step')
legend('Location','best')
grid on
%%
export_fig(fullfile(savedir,'fig_inv_gpinn_param.pdf'))

%%
close all

a = 2;
b = 4;

[ha, pos] = tight_subplot(1, 2, 0.1, 0.1, 0.1)

axes(ha(1))
hold on
plot(m(a).log.it, log10(m(a).log.res),'DisplayName','PINN res')
plot(m(a).log.it, log10(m(a).log.data),'DisplayName','PINN data')
plot(m(a).log.it, log10(m(a).log.total),'DisplayName','PINN total')
title('Losses of inverse PINN')

axes(ha(2))
hold on
plot(m(b).log.it, log10(m(b).log.res),'DisplayName','gPINN res')
plot(m(b).log.it, log10(m(b).log.data),'DisplayName','gPINN data')
plot(m(b).log.it, log10(m(b).log.total),'DisplayName','gPINN total')
plot(m(b).log.it, log10(m(b).log.dres* 1e-2),'DisplayName','gPINN 0.01\nablares')
title('Losses of inverse gPINN')

for i = 1:2
    axes(ha(i))
    grid on
    xlabel('step')
    ylabel('log10(loss)')
    ylim([-5.5 -0.5])
%     yticks([-5.5:0.5:-1])
     yticklabels('auto')
     xticklabels('auto')
    legend('Location','best')
end
%%
export_fig(fullfile(savedir,'fig_inv_gpinn_losses.pdf'))
