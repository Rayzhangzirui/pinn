classdef GliomaSolver<handle
    % model to solve glioma
    % if 2d model, all the properties are only slices
    % 
    
    properties
        % diffusion coeff and proliferation rate, with unit

        dw 
        dg
        rho
        x0 % initial location
        ix 
        dim %dimension
        tend % final time
        
        soldir % directory to save fdm solution

        df % diffusion coefficient as field

        name % unique string used to identify data
        
        zslice % slice if 2d, or slice for visual in 3d
        
        % ndgrid, used for interpolation
        gx
        gy
        gz

        % anatomy data
        Pwm
        Pgm
        Pcsf    
        
        % solution
        phi
        uall
        tall
        uend
        
        % scaling
        rmax % radius of tumor region
        L
        T
        DW
        RHO
        
        end
    
    methods
        function obj = GliomaSolver(dim, d, rho, x0, tend, zslice)
            obj.dim = dim;
            obj.dw = d;
            obj.dg = d/10;
            obj.rho = rho; %1/day
            obj.tend = tend; % day
            obj.x0 = x0;
            obj.zslice  = zslice;

            if obj.dim == 2
                obj.ix = [obj.x0 1];
            else
                obj.ix = obj.x0;
            end

            % if dim 2, use slice as identifier for z
            if dim==2
                ztmp = zslice;
            else
                ztmp = x0(3);
            end
            obj.name = sprintf("dim%d_dw%g_rho%g_ix%d_iy%d_iz%d_tend%d",...
                dim, d, rho, x0(1), x0(2), ztmp, tend);
            fprintf('model tag %s\n',obj.name);
            
        end
        
        function readmri(obj,fdir)
            % read mri data, fdir = dir to atlas

            gm = MRIread( [fdir 'GM.nii']);
            wm = MRIread( [fdir 'WM.nii']);
            csf = MRIread( [fdir 'CSF.nii']);
            
            obj.Pwm = wm.vol;
            obj.Pgm = gm.vol;
            obj.Pcsf = csf.vol;

            if obj.dim == 2
            %%% for 2d case, slice 3d MRI, then set zslice = 1, the third dimension is
            %%% just 1. Filter with periodic padding make all z-derivative 0. So same as 2D
                obj.Pwm  = obj.Pwm(:,:,obj.zslice);
                obj.Pgm  = obj.Pgm(:,:,obj.zslice);
                obj.Pcsf = obj.Pcsf(:,:,obj.zslice);
            end

            obj.df = obj.Pwm*obj.dw + obj.Pgm*obj.dg; % diffusion coefficients
            
            sz = [1 1 1];
            sz(1:obj.dim) = size(obj.Pwm); % even when Pwm is 2d, 3rd dim is 1
            [obj.gx,obj.gy,obj.gz] = ndgrid(1:sz(1),1:sz(2),1:sz(3)); 
        end% 
        
        function savesol(obj,fp)
            % save solution to mat file
            if isfile(fp)
                fprintf('%s already exist\n',fp);
                return
            end
            uall = obj.uall;
            tall = obj.tall;
            phi = obj.phi;
            save(fp,'uall','tall','phi','-v7.3');
            fprintf('save to %s\n',fp)
        end

        function loadsol(obj, fp)
            % load result of fdm solution, uall, tall, phi
            assert(isfile(fp), '%s not found\n',path);

            dat = load(fp);
            obj.uall = dat.uall;
            obj.tall = dat.tall;
            obj.phi = dat.phi;
            if obj.dim == 2
                obj.uend = obj.uall(:,:,end);
            else
                obj.uend = obj.uall(:,:,:,end);
            end
        end

        function [fp, es] = solpath(obj)
            fp = fullfile(obj.soldir, sprintf("sol_%s.mat",obj.name));
            es = isfile(fp);
        end

        function solve(obj)
            [fp,es] = obj.solpath;
            if es
                fprintf('load existing solution\n');
                obj.loadsol(fp);
            else
               fprintf('solve and save pde\n');
               [obj.phi,obj.uall,obj.tall,obj.uend] = GliomaFdmSolve(obj.Pwm, obj.Pgm, obj.Pcsf, obj.dw, obj.rho, obj.tend, obj.ix, obj.dim);
               obj.savesol(fp); 
            end

            obj.getrmax();
            
        end

        function f = getSlice(obj, name, vz, tk)
            % get data at time index tk, slice vz (optional)
            if nargin < 4
                tk = length(obj.tall); % tk defaul end
            end
            if nargin < 3
                vz = obj.zslice; % vz default zslice
            end

            dat = obj.(name);
            datdim = length(size(dat));
        
            if datdim==4
                f = dat(:,:,vz,tk);
            elseif datdim == 3
                f = dat(:,:,vz);
            else
                f = dat;
            end
        end

        function [fig,ax1] = plotbkgd(obj, dat)
            
            fig = figure;
            ax1 = axes;
            
            h1 = imagesc(ax1,dat);
            cmap = colormap(ax1,gray(20));
            cb1 = colorbar(ax1,'Location','westoutside');
        end
        
        function [fig,ax1, ax2, hLink] = twoimagesc(obj, bgdat, fgdat)
            
            [fig,ax1] = plotbkgd(obj, bgdat);
            
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

        function [fig, ax1, ax2] = imagesc(obj, bgname, fgdat, varargin)
            bgdat = obj.getSlice(bgname, varargin{:});
            [fig, ax1, ax2]  = obj.twoimagesc(bgdat, fgdat);
        end

        function [fig, ax1, ax2] = plotuend(obj,varargin)
            uend = obj.getSlice('uend',varargin{:});
            [fig, ax1, ax2]  = obj.imagesc('df', uend, varargin{:});
        end

        function getrmax(obj)
            % get radius of solution
            tk = length(obj.tall); 
            
            
            idx = find(obj.uend.*obj.phi>0.01); % index of u value greater than threshold
            
            distix = sqrt((obj.gx-obj.ix(1)).^2+ (obj.gy-obj.ix(2)).^2+(obj.gz-obj.ix(3)).^2);
            [maxdis,maxidx] = max(distix(idx));
            obj.rmax = maxdis+3;
            ls = distix - obj.rmax; % level set function
            fprintf('rmax = %g\n',obj.rmax);
        end

        function fq = interpf(obj, f, X, method)
            % interpolate f(x)
            if obj.dim == 3
                fq = interpn(obj.gx, obj.gy, obj.gz, f, X(:,1), X(:,2), X(:,3), method);
            else
                fq = interpn(obj.gx, obj.gy, f, X(:,1), X(:,2), method);
            end 
        end

        function fq = interpu(obj, t, X, method)
            % interpolate u(x,t)

            f = @(x) reshape(x,[],1);

            if obj.dim == 3
                xg = f(obj.gx(:,1,1));
                yg = f(obj.gy(1,:,1));
                zg = f(obj.gz(1,1,:));
                fq = interpn(xg, yg, zg, obj.tall, obj.uall, X(:,1), X(:,2), X(:,3), t, method);
            else
                xg = f(obj.gx(:,1));
                yg = f(obj.gy(1,:));
                fq = interpn(xg, yg, obj.tall, obj.uall, X(:,1), X(:,2), t, method);
            end 
        end

        function [fig,ax1,ax2,hLink] = scatter(obj,bgname,X,sz,dat,varargin)
            [fig,ax1] = plotbkgd(obj, bgname);
            ax2 = axes;
            scatter(ax2,X(:,2),X(:,1),sz,dat,varargin{:});
            set(ax2,'YDir','reverse')
            set(ax2,'color','none','visible','off');
            cb2 = colorbar(ax2,'Location','eastoutside');
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end

        function scale(obj, dwc, rhoc, lc)
            % scale the data for training
            % dwc, rhoc characteristic value with unit
            % lc = characteristic length
            obj.L = lc;
            obj.DW = sqrt(dwc/rhoc)/lc;
            obj.RHO = sqrt(rhoc/dwc)*lc;
            obj.T = obj.L/sqrt(dwc * rhoc);
        end
       
        function fp = ReadyDat(obj, n, varargin)
            % prepare data for training
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'noiseon', 'uqe'); %none = no noise, uq noise after interp, u interp after noise
            addParameter(p, 'method', 'linear');
            addParameter(p, 'seed', 1);
            addParameter(p, 'tag', '');
            addParameter(p, 'datdir', './'); % which directory to save
            parse(p, varargin{:});
            
            seed = p.Results.seed;
            datdir = p.Results.datdir;
            [status,msg,msgid] = mkdir(datdir);
            assert(status);
            method = p.Results.method;
            if ~isempty(p.Results.tag)
                tag = "_"+p.Results.tag;
            end

            % sample collocation points
            rng(seed,'twister');
            xq = sampleDenseBall(n, obj.dim, obj.L, obj.x0); % sample coord, unit
            tq = rand(n,1)*obj.tend; % sample t, unit

            Pwmq = obj.interpf(obj.Pwm, xq, method);
            Pgmq = obj.interpf(obj.Pgm, xq, method);
            phiq = obj.interpf(obj.phi, xq, method);
            uq = obj.interpu(tq, xq, method); % u(tq, xq)
            tqe = ones(n,1)*obj.tend; % all final time

            uqe = obj.interpf(obj.uend, xq, method); % u(t_end,xq), without noise
            
            if strcmpi(p.Results.noiseon, 'uqe')
                % interp then add noise
                fprintf('interp then add noise\n');
                nzuqe = addNoise(uqe, p.Unmatched); % noisy uqe
            elseif strcmpi(p.Results.noiseon, 'uend')
                % add noise to uend then interp
                fprintf('add noise then interp\n');
                nzuend = addNoise(obj.uend, p.Unmatched);
                nzuqe = obj.interpf(nzuend, xq, method); % u(t_end,xq)
            else
               error('unknown option') 
            end

            % sample
            xqcenter = xq - obj.x0;

            % single point data 
            xdat = [tqe/obj.T xqcenter/obj.L];
            udat = nzuqe;

            % testing data, all time
            xtest = [tq/obj.T xqcenter/obj.L];
            utest = uq;
            
            
            DW = obj.DW;
            RHO = obj.RHO;
            L = obj.L;
            T = obj.T;
            argsample = varargin;
            
            fp = sprintf('dat_%s_n%d%s.mat',obj.name,n,tag);
            fp = fullfile(datdir, fp);
            save(fp,'xdat','udat','uqe','xq','xtest','utest','DW','RHO','L','T', 'Pwmq', 'Pgmq', 'phiq','n','argsample','seed');
            fprintf('save training dat to %s\n', fp );
        end
        
        function [ts,x,upred,upredall] = loadresults(obj, predmatfile)
            % load neural net results
            s = load(predmatfile);
            x = double(s.x) * obj.L + obj.x0;
            ts = double(s.ts) * obj.T;
            upredall = double(s.upred);
            i = find(ts==obj.tend);
            upred = upredall(:,i);
        end

        function p = plotline(obj)
            % 1d plot of data, mid section
            u = obj.uend.*obj.phi;
            [~,mid] = getBox(u, 0.01, 0);
            
            y = u(:,mid(2));
            x = 1:length(y);
            p = plot(x,y,'k','DisplayName','u');
            legend(p,'Location','best');
            
        end


        
    end
end



