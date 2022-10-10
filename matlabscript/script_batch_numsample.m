% study number of samples
mdirs = '/Users/Ray/project/glioma/pinn/models/noise_uend_std/';
whichdir = '/Users/Ray/project/glioma/pinn/models/noise_uend_std/std*'
[~,modeldirs] = sdir(whichdir);
%%
for i = 1:length(modeldirs)
    modeldirpath = fullfile(modeldirs(i).folder, modeldirs(i).name);
    m(i) = PostData(modeldirpath,'tag', modeldirs(i).name);
    m(i).unscale(2, g.T, g.L, g.x0);
end
%%
mksz = 6;
alpha= 0.5;
for i = 1:length(modeldirs)
% for i = 1:1
    close all;
%     m(i).PlotLoss()
%     m(i).PlotInferErr(1.3, 1.25);
%     
    [fig,ax1,ax2] = g.scatter('df',m(i).upred{end}.xdat(:,2:end), mksz, m(i).upred{end}.upredxdat,'filled','MarkerFaceAlpha',alpha);
    ax2.CLim = [0 0.35];
    title(ax1,'u_{pred}')
    export_fig(fullfile(m(i).modeldir,'fig_upredxdat.jpg'),'-m3')
    
    % show final training data, (with noise)
    [fig,ax1,ax2] = g.scatter('df',m(i).upred{end}.xdat(:,2:end), mksz, m(i).upred{end}.udat,'filled','MarkerFaceAlpha',alpha);
    ax2.CLim = [0 0.35];
    title(ax1,'u_{noise dat}')
    export_fig(fullfile(m(i).modeldir,'fig_udat.jpg'),'-m3')
    
    % show final uend without noise
    [fig,ax1,ax2] = g.scatter('df',m(i).upred{end}.xdat(:,2:end), mksz, uqe,'filled','MarkerFaceAlpha',alpha);
    ax2.CLim = [0 0.35];
    title(ax1,'u_{end}')
    export_fig(fullfile(m(i).modeldir,'fig_uend.jpg'),'-m3')
    
    % show amount of noise
    [fig,ax1,ax2] = g.scatter('df',m(i).upred{end}.xdat(:,2:end), mksz, m(i).upred{end}.udat - uqe,'filled','MarkerFaceAlpha',alpha);
    title(ax1,'u_{noise dat} - u_{end}')
    export_fig(fullfile(m(i).modeldir,'fig_del_noise.jpg'),'-m3')
    
    % show err of upred
    [fig,ax1,ax2] = g.scatter('df',m(i).upred{end}.xdat(:,2:end), mksz, abs(m(i).upred{end}.upredxdat - uqe),'filled','MarkerFaceAlpha',alpha);
    title(ax1,'abs u_{pred} - u_{end}')
    export_fig(fullfile(m(i).modeldir,'fig_upred_err.jpg'),'-m3')
end

%%
ns = arrayfun(@(x) x.info.n_res_pts, m);
[ns,idx] = sort(ns);
m = m(idx);
%%
close all
k = 1;
errD = arrayfun(@(x) x.getRelErr(k,'rD',1.3), m);
errRHO = arrayfun(@(x) x.getRelErr(k,'rRHO',1.25), m);

close all
scatter(ns,abs(errD)*100,'filled','DisplayName','rD')
hold on
scatter(ns,abs(errRHO)*100,'filled','DisplayName','rRHO')
legend('Location','best')
xlabel('n pts')
ylabel('%rel err')
title('% rel err vs n (ADAM)')
grid on
% export_fig(fullfile(mdirs,'fig_err_npts_adam.jpg'),'-m3')

%%

rateadam = arrayfun(@(x) x.info.tfadamtime/x.info.tfadamiter*1000, m);
ratelbfgs = arrayfun(@(x) x.info.scipylbfgstime/x.info.scipylbfgsiter*1000, m);
close all
scatter(ns,rateadam,'filled','DisplayName','adam')
hold on
scatter(ns,ratelbfgs,'filled','DisplayName','lbfgs')
grid on
title('training speed')
legend('Location','best')
xlabel('n')
ylabel('sec per 1k steps')
export_fig(fullfile(mdirs,'fig_speed.jpg'),'-m3')