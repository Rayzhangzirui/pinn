datadir = '/Users/zziruiadmin/projects/glioma/pinndata/';
modelname = 'growth2d_inv';
%%
trainfile = fullfile(datadir,modelname,'history.dat');
t = readtable(trainfile);
t.Properties.VariableNames = {'it','lres','ldat','ltot','mres','D','RHO'};
sp = 100; % sampling period
t = t(1:sp:end,:);
%%
plot(t.it, log10(t{:,2:5}))
grid on
legend('lres','ldata','ltot','mxres')
xlabel('it')
ylabel('$$\log_{10}$$(loss)')
title('loss vs. iteration')
%%
export_fig(fullfile(datadir,modelname,'fig_loss.pdf'))
%% run pde to get D RHO

relerr = abs([t.D t.RHO] - [D RHO])./[D RHO];

plot(t.it, (relerr))
grid on
legend('D','\rho','Location','best')
xlabel('it')
ylabel('rel err')
title('rel err vs. iteration')
%%
export_fig(fullfile(datadir,modelname,'fig_relerr.pdf'))

%%
predfile = fullfile(datadir,modelname,'tpred.txt');
tp = readtable(predfile);
tp.r = sqrt(tp{:,1}.^2+tp{:,2}.^2);
%%
close all

tidx = [1 25 50 75 100]
lgs = {'t=0','t=0.25','t=0.5','t=0.75','t=1'}

sigma = 0.1;
nsol = sol + sigma*randn(size(sol));

fig = figure
ax = gca
hold(ax,'on')

for i = 1:length(tidx)
    j = tidx(i);
    p(j) = plot(ax,Rarray, sol(j,:),'DisplayName',lgs{i});
    s(j) = scatter(ax,tp.r, tp{:,2+i},12,'filled');
    s(j).MarkerFaceColor = p(j).Color;
    s(j).Annotation.LegendInformation.IconDisplayStyle = 'off';
end
ylim([0,1])
xlim([0,1.1])
legend('Location','best')
title('exact u vs pred u')
%%
export_fig(fullfile(datadir,modelname,'fig_pred.pdf'))
