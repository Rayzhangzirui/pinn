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
        
        function savesol(obj)
            % save solution to mat file
            fsol = sprintf('sol_%s',obj.name);
            fp = fullfile(fsol);
            
            
            yessave = 'n';
            issave = false;
            if isfile(fp)
                fprintf('%s already exist\n',datafname);
                yessave=input('Do you want to continue, y/n [y]:','s');
                if yessave~='y'
                    issave = true;
                end
            else
                issave = true;
            end
                
            if issave
                fprintf('save to %s\n',fp);
                save(fp,'obj.uall','obj.tall','obj.phi','-v7.3');
            else
                fprintf('not saved');
            end
            
        end

        function solve(obj)
            [obj.phi,obj.uall,obj.tall,obj.uend] = GliomaFdmSolve(obj.Pwm, obj.Pgm, obj.Pcsf, obj.dw, obj.rho, obj.tend, obj.ix, obj.dim);
            obj.getrmax();
            fprintf('rmax = %g\n',obj.rmax);
%             obj.savesol();
        end

        function [u, d, phi] = getdat(obj, tk, vz)
            % get data at time index tk, slice vz (optional)
            if obj.dim==3
                if nargin == 3
                    u = obj.uall(:,:,vz,tk);
                    d = obj.df(:,:,vz);
                    phi = obj.phi(:,:,vz);
                else
                    u = obj.uall(:,:,:,tk);
                    d = obj.df(:,:,:);
                    phi = obj.phi(:,:,:);
                end
            end

            if obj.dim == 2
                uslice = obj.uall(:,:,tk);
                d = obj.df;
                phi = obj.phi;
                u = obj.uall(:,:,tk);
            end
        end

        function [fig,ax1] = plotbkgd(obj, bgname)
            assert(obj.dim == 2); % todo: 3d
            dat = obj.(bgname);
            sz = size(dat);
            fig = figure;
            ax1 = axes;
            
            h1 = imagesc(ax1,dat);
%             axis equal;
%             xlim([1 sz(1)])
%             ylim([1 sz(2)])
            
            
            cmap = colormap(ax1,gray(20));
            cb1 = colorbar(ax1,'Location','westoutside');
        end
        
        function [fig,ax1, ax2, hLink] = imagesc(obj,bgname, varargin)
            [fig,ax1] = plotbkgd(obj, bgname);
            ax2 = axes;
            h2 = imagesc(varargin{:});
            cmp = colormap(ax2,'parula');
            h2.AlphaData = double(h2.CData>1e-2); % threshold for transparency
            maxcdata = max(h2.CData(:));
            clim(ax2,[0,maxcdata]);
            set(ax2,'color','none','visible','on')
            cb2 = colorbar(ax2,'Location','eastoutside')
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end

        function [fig, ax1, ax2] = plotuend(obj)
            [fig, ax1, ax2]  = obj.imagesc('df', obj.uend);
        end

        function getrmax(obj)
            % get radius of solution
            tk = length(obj.tall); 
            [u,d,phi] = obj.getdat(tk);
            
            idx = find(u.*phi>0.01); % index of u value greater than threshold
            
            distix = sqrt((obj.gx-obj.ix(1)).^2+ (obj.gy-obj.ix(2)).^2+(obj.gz-obj.ix(3)).^2);
            [maxdis,maxidx] = max(distix(idx));
            obj.rmax = maxdis+3;
            ls = distix - obj.rmax; % level set function
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
        
        function [h1,ax1,h2,ax2] = visualover(obj, prop, varargin)
            % visualize 2d scattered data, overlay
            
            bkgd = obj.(prop);
            
            fig = figure
            ax1 = axes;
            colormap(ax1,gray(20));
            h1 = imagesc(bkgd);
            cb1 = colorbar(ax1,'Location','westoutside');
            
            ax2 = axes;
            h2 = scatter(ax2, varargin{:});
            cmp = colormap(ax2,'parula');
            set(ax2,'YDir','reverse');
            % caxis(ax2,[minu maxu]); % caxis is clim in newer version of matlab
            set(ax2,'color','none','visible','off');
            cb2 = colorbar(ax2,'Location','eastoutside');
            % 
            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position','DataAspectRatio'});
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

        function datname = ReadyDat(obj, n, varargin)
            % prepare data for training
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'noiseon', 'uqe'); %none = no noise, uq noise after interp, u interp after noise
            addParameter(p, 'method', 'linear');
            addParameter(p, 'seed', 1);
            addParameter(p, 'tag', '');
            parse(p, varargin{:});
            
            seed = p.Results.seed;
            method = p.Results.method;
            if ~isempty(p.Results.tag)
                tag = "_"+p.Results.tag;
            end


            % sample collocation points
            rng(seed,'twister');
            xq = sampleDenseBall(n, obj.dim, obj.rmax, obj.x0); % sample coord, unit
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
                nzuqe = addNoise(uqe,varargin{:}); % noisy uqe
            elseif strcmpi(p.Results.noiseon, 'uend')
                % add noise to uend then interp
                fprintf('add noise then interp\n');
                nzuend = addNoise(obj.uend,varargin{:});
                nzuqe = obj.interpf(nzuend, xq, method); % u(t_end,xq)
            else
               error('unknow option') 
            end


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
            
            datname = sprintf('dat_%s_n%d%s.mat',obj.name,n,tag);
            save(datname,'xdat','udat','uqe','xq','xtest','utest','DW','RHO','L','T', 'Pwmq', 'Pgmq', 'phiq','n','argsample','seed');
            fprintf('save training dat to %s\n',datname)
        end
        
        function [ts,x,upred,upredall] = loadresults(obj, predmatfile)
            s = load(predmatfile);
            x = double(s.x) * obj.L + obj.x0;
            ts = double(s.ts) * obj.T;
            upredall = double(s.upred);
            i = find(ts==obj.tend);
            upred = upredall(:,i);
        end
    end
end


