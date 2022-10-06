function [s,l,hax] = plot_data(varargin)
% plot data in log-log scale and linear regression
% plot_data('xlabel',x,'y1name', y1, 'y2name', y2, ...)
% varargin = options in scatter


[ax, args, ~] = axescheck(varargin{:});

% Get handle to either the requested or a new axis.
if isempty(ax)
    hax = gca;
else
    hax = ax;
end

p = inputParser;
p.KeepUnmatched = true;
addRequired(p,'x');
addRequired(p,'y');
addParameter(p,'islog',true);
addParameter(p,'fitmethod','lsq');
addParameter(p,'name','');
parse(p,args{:});
x = p.Results.x;
y = p.Results.y;
islog = p.Results.islog;
fitmethod = p.Results.fitmethod;
name = p.Results.name;
    

% use different marker
mks = 'o+*xsd^v';
used_mks = '';
h = get(ax,'children');
for i = 1:length(h)
    if strcmp(get(h(i),'type'),'scatter')
        mk = get(h(i),'Marker'); % sometime return 'square'
        used_mks(end+1) = mk(1);
    end
end
unused_mks = setdiff(mks,used_mks,'stable');
if isempty(unused_mks)
    nxt_mk = mks(randi([1,length(mks)]));
else
    nxt_mk = unused_mks(1);
end



% valid index
valid = @(v) find(~isnan(v) & ~isinf(v));
vidx = valid(y);
x = x(vidx);
y = y(vidx);


if islog
    x = log10(x);
    y = log10(y);
end

% fit
if strcmp(fitmethod,'lsq')
    pcoef = polyfit(x,y,1);
    f = polyval(pcoef,x); 
    m = pcoef(1);
    fprintf('use polyfit, intercep = %f, slope = %f.\n',pcoef(2),pcoef(1));
else
    
    b = robustfit(x,y);
    f = b(1) + b(2)*x;
    m = b(2);
    fprintf('use robustfit, intercep = %f, slope = %f.\n',b(1),b(2));
end


%     eqn = ['y = ' sprintf('%.2f x + %.2f',p)];
%     text(ax,logx(end),f(end),eqn);
hold(hax,'on')
s = scatter(hax,x,y, nxt_mk,'DisplayName',sprintf('%s: m = %.2f',name,m));
l = plot(hax,x,f,'Color',get(s,'CData'));

set(get(get(l,'Annotation'),'LegendInformation'),'IconDisplayStyle','off'); 
legend(hax,'Location','best');
% to skip legend of fitted line
% https://www.mathworks.com/matlabcentral/answers/406-how-do-i-skip-items-in-a-legend

% set(findall(gcf,'-property','FontSize'),'FontSize',14);
% set(findall(gcf,'-property','LineWidth'),'LineWidth',1.5);


end