classdef Atlas<DataSet
    % NOT READY
    % atlas of brain geometry

    methods
        function obj = Atlas(varargin)
            p.fdir = '/Users/Ray/project/glioma/jana/Atlas/anatomy/';
            p.zslice = 100;
            p.xdim = 2;
            p.dw = 0.1;
            p = parseargs(p, varargin{:});

            if strcmp(p.fdir,'sphere')
                fprintf('use 2d sphere \n');
                n = 201;
                R = 75;
                mid = ceil(n/2);
                [gx,gy] = ndgrid(1:n,1:n); 
    
                distix = sqrt((gx-mid).^2+ (gy-mid).^2);
                Pwm = double(distix<R);
                Pgm = zeros(n);
                Pcsf = zeros(n);
            else
                gm = MRIread( [p.fdir 'GM.nii']);
                wm = MRIread( [p.fdir 'WM.nii']);
                csf = MRIread( [p.fdir 'CSF.nii']);
                
                Pwm = wm.vol;
                Pgm = gm.vol;
                Pcsf = csf.vol;
                
                if p.xdim == 2
                %%% for 2d case, slice 3d MRI, then set zslice = 1, the third dimension is
                %%% just 1. Filter with periodic padding make all z-derivative 0. So same as 2D
                    Pwm  = Pwm(:,:,p.zslice);
                    Pgm  = Pgm(:,:,p.zslice);
                    Pcsf = Pcsf(:,:,p.zslice);
                end
            end

            obj@DataSet(Pwm, Pgm, Pcsf)
            obj.setdw(p.dw)
        end

        function setdw(obj, dw)
            dg = dw/10;
            df = obj.Pwm*dw + obj.Pgm*dg; % diffusion coefficients
            obj.addvar(df,dw,dg);
        end


        function [ax1, h1] = plotbkgd(obj, dat)
            h1 = imagesc(dat);
            ax1 = h1.Parent;
            cmap = colormap(ax1,gray(20));
            cb1 = colorbar(ax1,'Location','westoutside');
        end
        
        function [ax1, ax2] = imagesc2(obj, bgname, fgdat, varargin)
            bgdat = slice2d(obj.(bgname), varargin{:});
            fgdat = slice2d(fgdat, varargin{:});
            
            [ax1,h1] = plotbkgd(obj, bgdat);
            ax1.Position(3) = ax1.Position(3)-0.1;
            
            ax2 = axes;
            h2 = imagesc(fgdat);
            cmp = colormap(ax2,'parula');
            h2.AlphaData = double(h2.CData>1e-2); % threshold for transparency
            maxcdata = max(h2.CData(:));
            clim(ax2,[0,maxcdata]);
            set(ax2,'color','none','visible','on')
            cb2 = colorbar(ax2,'Location','eastoutside')
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end


        function [fig,ax1,ax2,hLink] = scatter(obj,bgname,X,varargin)
            % backgound imagesc is plotted with YDir reversed
            % grid is ndgrid, so x coor is vertical, y coord is horizontal
            bgdat = slice2d(obj.(bgname), varargin{:});
            [fig,ax1] = plotbkgd(obj, bgdat);
            ax1.Position(3) = ax1.Position(3)-0.1;
            ax2 = axes;
            scatter(ax2,X(:,2),X(:,1),varargin{:});
            set(ax2,'YDir','reverse')
            set(ax2,'color','none','visible','off');
            cb2 = colorbar(ax2,'Location','eastoutside');
            
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end
 





        

    end
end