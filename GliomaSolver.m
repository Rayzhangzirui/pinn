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
        
        seed % random seed, sampling

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
        
        % radius of tumor region
        rmax 
        L
        T
        DW
        RHO

        % sample
        Pwmq
        Pgmq
        phiq
        dq
        xq
        tq
        n % number of spatial points
        uq
        uqe % uq at final time
        
       
        datadir = '/Users/Ray/project/glioma/matlabscripts/data/';
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
            % read mri data
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
            fsol = sprintf('sol_%s',obj.name);
            fp = fullfile(obj.datadir, fsol );
            
            
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
            [obj.phi,obj.uall,obj.tall] = GliomaFdmSolve(obj.Pwm, obj.Pgm, obj.Pcsf, obj.dw, obj.rho, obj.tend, obj.ix, obj.dim);
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

        function [ax1, h1, ax2, h2] = plotu(obj, varargin)
            % tk = which time step to visualize
            % slice = which slice
            % bg = background D or phi
            % th = threshold for transparent u
            p = inputParser;
            addParameter(p,'tk',length(obj.tall));
            addParameter(p,'slice', obj.zslice);
            addParameter(p,'bg', 'D');
            addParameter(p,'th', 0.01);
            
            parse(p,varargin{:});
            tk = p.Results.tk;
            vz = p.Results.slice;
            th = p.Results.th;
            bg = p.Results.bg;
            
            [uslice, Dslice, phislice] = obj.getdat( tk, vz);
            
            if strcmp(bg,'D')
                background = Dslice;
            elseif strcmp(bg,'phi')
                background = phislice;
            else 
                error('unknow backgroun')
            end
            
            maxu = max(obj.uall(:)); % maximum concentration, used to scale color map

            ax1 = axes;
            colormap(ax1,gray(20));
            h1 = imagesc(background);
            cb1 = colorbar(ax1,'Location','westoutside');

            ax2 = axes;
            h2 = imagesc(uslice.*phislice);
            h2.AlphaData = double(uslice>th); % threshold for transparency
            cmp = colormap(ax2,'parula');
            caxis(ax2,[0 maxu]); % caxis is clim in newer version of matlab
            set(ax2,'color','none','visible','on')
            cb2 = colorbar(ax2,'Location','eastoutside')

            hLink = linkprop([ax1,ax2],{'XLim','YLim','Position'});
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

        function sample(obj,n,seed)
            if nargin == 2
                obj.seed = 1;
            end
            % sample collocation points
            obj.getrmax();
            obj.n = n;

            %% interpolate the data
            rng(obj.seed,'twister');
            obj.xq = sampleDenseBall(n,obj.dim, obj.rmax, obj.x0); % sample coord, unit
            obj.tq = rand(n,1)*obj.tend; % sample t, unit
        end

        function fq = interpf(obj, f, method)
            % interpolate in ndgrid
            if obj.dim == 3
                fq = interpn(obj.gx, obj.gy, obj.gz, f, obj.xq(:,1), obj.xq(:,2), obj.xq(:,3),method);
            else
                fq = interpn(obj.gx, obj.gy, f, obj.xq(:,1), obj.xq(:,2) ,method);
            end 
        end

        function fq = interpu(obj, t, method)
            f = @(x) reshape(x,[],1);
            % interpolate  u ndgrid and t
            if obj.dim == 3
                xg = f(obj.gx(:,1,1));
                yg = f(obj.gy(1,:,1));
                zg = f(obj.gz(1,1,:));
                fq = interpn(xg, yg, zg, obj.tall, obj.uall, obj.xq(:,1), obj.xq(:,2), obj.xq(:,3), t, method);
            else
                xg = f(obj.gx(:,1));
                yg = f(obj.gy(1,:));
                fq = interpn(xg, yg, obj.tall, obj.uall, obj.xq(:,1), obj.xq(:,2), t, method);
            end 
        end

        function interp(obj,method)
            obj.Pwmq = obj.interpf(obj.Pwm, method);
            obj.Pgmq = obj.interpf(obj.Pgm, method);
            obj.phiq = obj.interpf(obj.phi, method);
            obj.uq = obj.interpu(obj.tq, method);
            obj.uqe = obj.interpu(ones(obj.n,1)*obj.tend, method);
        end

        function hsc = visualxq(obj,prop)
            % visualize sampled points
            dat = obj.(prop);
            if obj.dim == 2
                hsc = scatter(obj.xq(:,2),obj.xq(:,1),6,dat,'filled');
                set(gca,'YDir','reverse')
            end

            if obj.dim == 3
                hsc = scatter3(obj.xq(:,2),obj.xq(:,1),obj.xq(:,3),6,dat,'filled');
                set(gca,'YDir','reverse')
            end
        end
        
        function [h1,h2] = visualover(obj, prop, varargin)
            % visualize 2d scattered data, overlay
            
            bkgd = obj.(prop);
            
            fig = figure
            ax1 = axes;
            colormap(ax1,gray(20));
            h1 = imagesc(bkgd);
            cb1 = colorbar(ax1,'Location','westoutside');
            
            ax2 = axes;
            h2 =scatter(ax2, varargin{:});
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
        
        function readydat(obj)
            % anatomy data
            Pwmq = obj.Pwmq;
            Pgmq = obj.Pgmq;
            phiq = obj.phiq;
            
            xqcenter = obj.xq - obj.x0;

            % single point data 
            xdat = [ones(obj.n,1) * obj.tend/obj.T xqcenter/obj.L];
            udat = obj.uqe;
            % testing data, all time
            xtest = [obj.tq/obj.T xqcenter/obj.L];
            utest = obj.uq;
            
            
            DW = obj.DW;
            RHO = obj.RHO;
            L = obj.L;
            T = obj.T;
            
            datname = sprintf('dat_%s.mat',obj.name);
            save(datname,'xdat','udat','xtest','utest','DW','RHO','L','T', 'Pwmq', 'Pgmq', 'phiq');
        end

    end
end



