classdef Atlas<DataSet
    % NOT READY
    % atlas of brain geometry

    methods
        function obj = Atlas(varargin)
            p.fdir = '/Users/Ray/project/glioma/jana/Atlas/anatomy/';
            p.zslice = 100;
            p.xdim = 2;
            p.dw = 0.1;
            p.wfilt = 1;
            
            p = parseargs(p, varargin{:});

            if strcmp(p.fdir,'sphere')
                p.bd = 60; % boudnary
                p.Rwm = 50; % radius of wm region
                p.Rgm = 0; % radius of gm region, if <p.Rwm, no gm
                
                p = parseargs(p, varargin{:});
                
                fprintf('use 2d sphere with radius %g, box %g \n',p.Rwm, p.bd);
                mid = 0;
                xgrid = -p.bd:p.bd;
                [gx,gy,gz] = ndgrid(xgrid,xgrid,1:1); 
    
                distix = sqrt((gx-mid).^2+ (gy-mid).^2);
                
                % smooth transition
                w = 1;
                shv = @(x,r) 1/2* (1 + tanh((r - x)/w)); 
                Pwm = shv(distix,p.Rwm);
                Pgm = (1-shv(distix,p.Rwm)).*(shv(distix,p.Rgm));

                % sharp transition
%                 Pwm = double(distix<p.Rwm);
%                 Pgm = double(distix>=p.Rwm).*double(distix<p.Rgm);
                Pcsf = zeros(size(Pwm));
            else
                fprintf('read atlas %s, slice %g\n', p.fdir, p.zslice);
                gm = MRIread( [p.fdir 'GM.nii']);
                wm = MRIread( [p.fdir 'WM.nii']);
                csf = MRIread( [p.fdir 'CSF.nii']);
                
                Pwm = wm.vol;
                Pgm = gm.vol;
                Pcsf = csf.vol;
                
                if p.xdim == 2
                fprintf('slice atlas z %d\n', p.zslice);
                %%% for 2d case, slice 3d MRI, then set zslice = 1, the third dimension is
                %%% just 1. Filter with periodic padding make all z-derivative 0. So same as 2D
                    Pwm  = Pwm(:,:,p.zslice);
                    Pgm  = Pgm(:,:,p.zslice);
                    Pcsf = Pcsf(:,:,p.zslice);
                end

                sz = [1 1 1];
                sz(1:p.xdim) = size(Pwm); % even when Pwm is 2d, 3rd dim is 1
                [gx,gy,gz] = ndgrid(1:sz(1),1:sz(2),1:sz(3)); 
            end

            if p.wfilt > 0
                fprintf('gauss filter sig = %d\n', p.wfilt);
                f = @(x) imgaussfilt(x, p.wfilt,'FilterDomain','spatial');
                [Pcsf, Pwm, Pgm] = mapfun(f, Pcsf, Pwm, Pgm);
            end
            
            obj@DataSet(Pwm, Pgm, Pcsf, gx, gy, gz);
            obj.setdw(p.dw, 10);

            
        end

        function setdw(obj, dw, factor)
            fprintf('compute diffusion field with dw = %g\n',dw);
            dg = dw/factor;
            df = obj.Pwm*dw + obj.Pgm*dg; % diffusion coefficients
            obj.addvar(df,dw,dg);
        end


        function [ax1, h1] = plotbkgd(obj, dat)
            if ischar(dat)
                dat = obj.(dat);
            end
                ax1 = axes;
            h1 = imagesc(obj.gx(1,1),obj.gy(1,1),dat);
%             maxcdata = max(h1.CData(:));
%             mincdata = min(h1.CData(:));
%             clim(ax1,[mincdata,maxcdata]);
            cmap = colormap(ax1,gray(20));
            cb1 = colorbar(ax1,'Location','westoutside');
        end
        
        function [ax1, ax2] = imagescfg(obj, fgdat, varargin)
            p.bgname = 'df';
            p.th = 0.1;
            p = parseargs(p, varargin{:});
            
            bgdat = slice2d(obj.(p.bgname), varargin{:});
            fgdat = slice2d(fgdat, varargin{:});
            
            [ax1,h1] = plotbkgd(obj, bgdat);
            ax1.Position(3) = ax1.Position(3)-0.1;
            
            ax2 = axes;
            h2 = imagesc(obj.gx(1),obj.gy(1),fgdat);
            cmp = colormap(ax2,'parula');
            h2.AlphaData = double(h2.CData>p.th); % threshold for transparency
            maxcdata = max(h2.CData(:));
            clim(ax2,[0,maxcdata]);
            set(ax2,'color','none','visible','on')
            cb2 = colorbar(ax2,'Location','eastoutside')
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
            
        end

        function [ax1, ax2, c] = contour(obj, bgname, fgdat, level, varargin)
            figure;
            bgdat = slice2d(obj.(bgname), varargin{:});
            fgdat = slice2d(fgdat, varargin{:});
            
            [ax1,h1] = plotbkgd(obj, bgdat);
            ax1.Position(3) = ax1.Position(3)-0.1;
            
            ax2 = axes;
            [~,c] = contour(obj.gy, obj.gx, fgdat, level,'b','LineWidth',2);
            set(ax2,'YDir','reverse')
            set(ax2,'color','none','visible','on')
            cb2 = colorbar(ax2,'Location','eastoutside')
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end

        function [ax1, ax2, c, uq] = contoursc(obj, X, u, level)
            % contour from scattered, extrapolate to grid
            X = double(X);
            u = double(u);
            F = scatteredInterpolant(X(:,2), X(:,3), u, 'linear','none');
            uq = F(obj.gx, obj.gy);
            uq(isnan(uq)) = 0;

            [ax1, ax2, c] = obj.contour('df',uq, level);
            
        end

        function [ax1, ax2] = imagescScatter(obj, X, u, varargin)
            % contour from scattered, extrapolate to grid
            X = double(X);
            u = double(u);
            F = scatteredInterpolant(X(:,2), X(:,3), u, 'linear','none');
            uq = F(obj.gx, obj.gy);
            [ax1, ax2] = obj.imagescfg(uq,'bgname','df',varargin{:});
        end


        function [ax1,ax2] = scatter(obj,bgname,X,varargin)
            % backgound imagesc is plotted with YDir reversed
            % grid is ndgrid, so x coor is vertical, y coord is horizontal
            figure;
            bgdat = slice2d(obj.(bgname));
            [ax1,~] = plotbkgd(obj, bgdat);
            ax1.Position(3) = ax1.Position(3)-0.1;
            ax2 = axes;
            scatter(ax2,X(:,end),X(:,end-1),varargin{:});
            set(ax2,'YDir','reverse')
            set(ax2,'color','none','visible','off');
            cb2 = colorbar(ax2,'Location','eastoutside');
            
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            
            hLink.Targets(1).DataAspectRatio = [1 1 1];
            
        end
 
        function getphi(obj,varargin)
            phi = solvePhaseField(obj.Pwm, obj.Pgm, obj.Pcsf, varargin{:});
            obj.addvar(phi);
        end



        

    end
end