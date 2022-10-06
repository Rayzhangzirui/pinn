% visualize error of results for gPINN
run startup.m
fudat = projpath('models/growth2d_xgrad/exactu_dim2_n20000.txt')
savedir = '/Users/zziruiadmin/projects/glioma/meeting/pinn20220817'

dirprefix = 'models/growth2d_xgrad/';
dirs = {'growth2d_fwd','growth2d_inv','growth2d_fwd_xgrad','growth2d_inv_xgrad'};
tags = {'foward-PINN','inv-PINN','forward-gPINN','inv-gPINN'};

for i = 1:length(dirs)
    p = fullfile(projdir,dirprefix,dirs{i})
    m(i) = Model(p,'tag',tags{i},'udat',fudat);
end

%%
xdim =  2;
ts = [0:0.25:1.25];
[sol,r] = polar_udat(ts,xdim);
%%
[ha, pos] = tight_subplot(1, 2, 0.1, 0.1, 0.1)
cols = [3:8]; % columns to plot
ms = [1 3]; % list of models
mse = @(x,y) mean((x-y).^2)
for i = 1:2
    axes(ha(i));
    hold on
    k = ms(i); % model 
    t = m(k).predt;
    t.r = sqrt(t{:,1}.^2 + t{:,2}.^2);
    
    t = t(t.r<1.1,:);
    for j = 1:length(cols)
        pred = t{:,cols(j)};
        interpu = interp1(r, sol(j,:), t.r);
        err = mse(pred, interpu);
        
        l(j) = plot(r, sol(j,:),'-k','LineWidth',2);

        sc(j) = scatter(t.r, pred, 6,'filled', 'DisplayName', sprintf('t=%-6.2f, mse=%.3e',ts(j),err));
        
    end
    xlabel('r')
    ylabel('u')
    ylim([-0.1,1])
    xlim([0,1.1])
    
    yticklabels('auto')
    xticklabels('auto')
    legend(sc, 'Location','best')
    grid on
    title(tags{k})
    

end
%%
export_fig(fullfile(savedir,'fig_fwd_gpinn_pred.pdf'))