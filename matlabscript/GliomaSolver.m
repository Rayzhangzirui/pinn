classdef GliomaSolver< dynamicprops
    % model to solve glioma
    % if 2d model, all the properties are only slices
    % 
    
    properties
        % diffusion coeff and proliferation rate, with unit

        dw 
        dg
        factor
        rho
        x0 % initial location
        ix 
        xdim %dimension
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
        atlas 
        
        % solution
        fdmsol
        
        % polar solution
        polarsol
        
        % scaling
        rmax % radius of tumor region
        dwc
        rhoc
        L
        T
        DW
        RHO
        rDe
        rRHOe
        
        end
    
    methods
        function obj = GliomaSolver(xdim, d, rho, x0, tend, zslice)
            % set up parameters
            obj.xdim = xdim;
            obj.dw = d;
            obj.factor = 10;
            obj.dg = d/obj.factor;
            obj.rho = rho; %1/day
            obj.tend = tend; % day
            obj.x0 = x0;
            obj.zslice  = zslice;

            if obj.xdim == 2
                obj.ix = [obj.x0 1];
            else
                obj.ix = obj.x0;
            end

            % if dim 2, use slice as identifier for z
            if xdim==2
                ztmp = zslice;
            else
                ztmp = x0(3);
            end
            obj.name = sprintf("dim%d_dw%g_rho%g_ix%d_iy%d_iz%d_tend%d",...
                xdim, d, rho, x0(1), x0(2), ztmp, tend);
            fprintf('model tag %s\n',obj.name);
            
        end

        function readmri(obj,varargin)
            % read mri data, fdir = dir to atla
            if length(varargin)==1 && isa(varargin{1},'Atlas')
                obj.atlas = varargin{1};
            else
                obj.atlas = Atlas('dw', obj.dw, 'zslice',obj.zslice, varargin{:});
            end

            [obj.gx,obj.gy,obj.gz] = deal(obj.atlas.gx, obj.atlas.gy, obj.atlas.gz); 
        end 
        
        function savesol(obj,fp)
            % save solution to mat file
            if isfile(fp)
                fprintf('%s already exist\n',fp);
                return
            end
            fdmsol = obj.fdmsol;
            save(fp,'-struct','fdmsol','-v7.3');
            fprintf('save to %s\n',fp)
        end

        function loadsol(obj, fp)
            % load result of fdm solution, uall, tall, phi
            assert(isfile(fp));

            obj.fdmsol = load(fp);
            obj.atlas.add('phi',obj.fdmsol.phi);
            if obj.xdim == 2
                obj.fdmsol.uend = obj.fdmsol.uall(:,:,end);
            else
                obj.fdmsol.uend = obj.fdmsol.uall(:,:,:,end);
            end
        end

        function [fp, es] = solpath(obj)
            fp = fullfile(obj.soldir, sprintf("sol_%s.mat",obj.name));
            es = isfile(fp);
        end

        function solve(obj,varargin)
            % solve pde, 
            p.redo = false;
            p.savesol = true;
            [p,unmatch] = parseargs(p, varargin{:});

            [fp,es] = obj.solpath;
            if es && p.redo == false
                fprintf('load existing solution\n');
                obj.loadsol(fp);
            else
                if ~isprop(obj.atlas,'phi')
                    fprintf('phi not computed, compute phi\n');
                    obj.atlas.getphi();
                end
                
                obj.fdmsol = GliomaFdmSolve(obj.atlas, obj.rho, obj.tend, obj.ix, unmatch{:});
                
                if p.savesol
                    fprintf('save pde solution\n');
                    obj.savesol(fp); 
                end
            end



            obj.getrmax();
        end

        function solvepolar(obj,varargin)
            fac = obj.factor; % factor = dw/dg
            p.epsilon = 1;
            p.bd = 50;
            p.m = obj.xdim-1;
            p.Rwm = p.bd;
            p.Rgm = p.bd*2;
            icfun = @(r) 0.1*exp(-0.1*r.^2);
            p.dx = 0.1;
            p.dt = obj.tend/fac;
            
            p = parseargs(p, varargin{:});
            disp(p);
            
            m = obj.xdim-1;
            
            % smooth transition
            w = p.epsilon;
            shv = @(x,r) 1/2* (1 + tanh((r - x)/w)); %smooth heaviside
            Dxshv = @(x,r) 0.5*(1-tanh((r - x)/w).^2)*(-1/w); % dx of smooth heavisdie function
            
            phifcn = @(x) shv(x, p.bd); 
            Pfcn = @(x) shv(x,p.Rwm)  + (1-shv(x,p.Rwm)).*(shv(x,p.Rgm))/fac;
            dffcn = @(x) obj.dw * Pfcn(x); % diffusion coefficent with unit function


            DxPfcn = @(x) Dxshv(x,p.Rwm)  - Dxshv(x,p.Rwm).*shv(x,p.Rgm)/fac + (1-shv(x,p.Rwm)).*Dxshv(x,p.Rgm)/fac;
            Dxphi = @(x) Dxshv(x,p.bd);
            DxPphi = @(x) phifcn(x).*DxPfcn(x) + Dxphi(x).*Pfcn(x);


            polargeo.Pwm = @(x) shv(x,p.Rwm);
            polargeo.Pgm = @(x) (1-shv(x,p.Rwm)).*(shv(x,p.Rgm));
            polargeo.P = Pfcn;
            polargeo.DxPphi = DxPphi;
            polargeo.phi = phifcn;


            addprop(obj,'polargeo');
            obj.polargeo = polargeo;

            % sharp transition
            % dwfcn = @(x) double(x<p.Rwm)*obj.dw + double(x>p.Rwm)*double(x<p.Rgm)*obj.dw/10;

            obj.polarsol = pdepolar(true, p.epsilon, p.bd, m, dffcn, obj.rho, p.dx, obj.tend, p.dt, icfun);
        end

        function fxi = interpGridPolar(obj, r, f)
            % interpolate f(r) on x, extrapolate as 0
            warning('interp polar assuming 2d');
            rxi = sqrt((obj.gx-obj.x0(1)).^2+(obj.gy-obj.x0(2)).^2);
            fri = interp1(r, f, rxi(:) );
            fri(isnan(fri)) = 0;
            fxi = reshape(fri, size(obj.gx));
        end



        function [vx,vy] = evalVectorPolar(obj,dfdr,X,Y)
            [theta,r] = cart2pol(X,Y);
            dr = dfdr(r);
            vx = dr.*X./r;
            vy = dr.*Y./r;
            vx(r<1e-3) = 0;
            vy(r<1e-3) = 0;
        end


        function interpPolarSol(obj)
            % interpolate polar solution to grid
            % make polar solution grid solution
            obj.fdmsol.tall = obj.polarsol.tgrid;
            warning('interp polar solution to grid, assuming xdim = 2');
            for i = 1:length(obj.fdmsol.tall)
                obj.fdmsol.uall(:,:,i) = obj.interpGridPolar(obj.polarsol.xgrid, obj.polarsol.sol(i,:));
            end
            obj.atlas.phi = obj.interpGridPolar(obj.polarsol.xgrid, obj.polarsol.phi);
            obj.atlas.P = obj.interpGridPolar(obj.polarsol.xgrid, obj.polarsol.df)/obj.dw;
            obj.fdmsol.uend = obj.fdmsol.uall(:,:,end);
                        
            [obj.fdmsol.DxPphi,obj.fdmsol.DyPphi] = obj.evalVectorPolar(obj.polargeo.DxPphi, obj.gx-obj.x0(1), obj.gy-obj.x0(2));
            obj.fdmsol.DzPphi = zeros(size(obj.fdmsol.DyPphi));

            obj.getrmax()
        end

        function f = getSlice(obj, name, vz, tk)
            % get data at time index tk, slice vz (optional)
            if nargin < 4
                tk = length(obj.fdmsol.tall); % tk defaul end
            end
            if nargin < 3
                vz = obj.zslice; % vz default zslice
            end

            dat = obj.(name);
            datdim = length(size(dat));
        
            if datdim==4
                f = dat(:,:,vz,tk);
            elseif datdim == 3 
                if obj.xdim == 3
                    f = dat(:,:,vz);
                else
                    f = dat(:,:,tk);
                end
            else
                f = dat;
            end
        end

        function [fig, ax1, ax2] = plotuend(obj,varargin)
            figure;
            uend = slice2d(obj.fdmsol.uend,varargin{:});
            [ax1, ax2]  = obj.atlas.imagescfg(uend, varargin{:});
        end

        function [fig, ax1, ax2] = plotphiuend(obj,varargin)
            fig = figure;
            uend = slice2d(obj.fdmsol.uend,varargin{:});
            phi = slice2d(obj.atlas.phi,varargin{:});
            [ax1, ax2]  = obj.atlas.imagescfg(uend.*phi, varargin{:});
        end

        function [rmax, idxin, mskin] = getrmax(obj,varargin)
            % get radius of solution
            p.padding = 0;
            p.threshold = 0.01;
            p = parseargs(p,varargin{:});
            
            padding = 1; % padding ot max distance
            threshold = 0.01; % threshold for tumor region


            idx = find(obj.fdmsol.uend.*obj.atlas.phi>threshold); % index of u value greater than threshold
            
            distix = sqrt((obj.atlas.gx-obj.ix(1)).^2+ (obj.atlas.gy-obj.ix(2)).^2+(obj.atlas.gz-obj.ix(3)).^2);
            [maxdis,maxidx] = max(distix(idx));
            rmax  = maxdis + padding;
            obj.rmax = rmax;

            lsf = distix - obj.rmax; % level set function
            mskin = lsf<0;
            idxin = find(lsf<0);
            fprintf('rmax = %g, padding = %g, threshod = %g\n',obj.rmax, p.padding, p.threshold);
        end

        function fq = interpf(obj, f, X, method)
            % interpolate f(x)
            if obj.xdim == 3
                fq = interpn(obj.gx, obj.gy, obj.gz, f, X(:,1), X(:,2), X(:,3), method);
            else
                fq = interpn(obj.gx, obj.gy, f, X(:,1), X(:,2), method);
            end 
        end

        function fq = interpu(obj, t, X, method)
            % interpolate u(x,t)
            f = @(x) reshape(x,[],1);

            if obj.xdim == 3
                xg = f(obj.gx(:,1,1));
                yg = f(obj.gy(1,:,1));
                zg = f(obj.gz(1,1,:));
                fq = interpn(xg, yg, zg, obj.fdmsol.tall, obj.fdmsol.uall, X(:,1), X(:,2), X(:,3), t, method);
            else
                xg = f(obj.gx(:,1));
                yg = f(obj.gy(1,:));
                fq = interpn(xg, yg, obj.fdmsol.tall, obj.fdmsol.uall, X(:,1), X(:,2), t, method);
            end 
        end

        function uq = interpgrid(obj, tq, method)
            % interpolate u(x,t)

            f = @(x) reshape(x,[],1);
            extrap = 'none';
            if obj.xdim == 3
                xg = f(obj.gx(:,1,1));
                yg = f(obj.gy(1,:,1));
                zg = f(obj.gz(1,1,:));
                tg = f(obj.fdmsol.tall);
                F = griddedInterpolant({xg, yg, zg, tg}, obj.fdmsol.uall, method,extrap);
                uq = F({xg,yg,zg,tq});
            else
                xg = f(obj.gx(:,1));
                yg = f(obj.gy(1,:));
                tg = f(obj.fdmsol.tall);
                F = griddedInterpolant({xg, yg, tg}, obj.fdmsol.uall, method,extrap);
                uq = F({xg,yg,tq});
            end 
            uq(isnan(uq)) = 0;
        end

        function scale(obj, dwc, rhoc, lc)
            % scale the data for training
            % dwc, rhoc characteristic value with unit
            % lc = characteristic length
            obj.dwc = dwc;
            obj.rhoc = rhoc;
            obj.L = lc;
            obj.T = obj.L/sqrt(dwc * rhoc);
            
            % characteristic dc*T, rhoc*T
            obj.DW = sqrt(dwc/rhoc)/lc;
            obj.RHO = sqrt(rhoc/dwc)*lc;
            
            % exact ratio
            obj.rDe = obj.tend/obj.T * obj.dw/obj.dwc;
            obj.rRHOe = obj.tend/obj.T * obj.rho/obj.rhoc;
            fprintf('T = %g, rDe = %g, rRHOe = %g\n', obj.T, obj.rDe, obj.rRHOe);
        end
        

        function [uq, xq, tq, phiq] = genGridDataUt(obj,mskin,ts)
            % generate final time data from FD grid
            if nargin < 3
                ts = length(obj.fdmsol.tall);
            end
            
            assert(obj.xdim == 2,'only in 2d');
            
            xq = [];
            phiq = [];
            uq = [];
            tq = [];
            for i = 1:length(ts)
                tk = ts(i);
                fprintf('sample grid data at t = %g\n',obj.fdmsol.tall(tk));
                xq = [xq;[obj.gx(mskin) obj.gy(mskin)]];
                phiq = [phiq;obj.atlas.phi(mskin)];
                u = obj.fdmsol.uall(:,:,tk);
                uq = [uq; u(mskin)];
                tq = [tq; ones(length(uq),1) * obj.fdmsol.tall(tk)];
            end

            p = randperm(length(uq));
            [uq, xq, tq, phiq] = mapfun(@(x) x(p,:),uq, xq, tq, phiq);

        end


        function [uq, xq, tq, Pwmq, Pgmq, phiq] = genGridDataUend(obj,mskin)
            % final time grid data
            assert(obj.xdim == 2,'only in 2d');
            xgrid = [obj.gx(mskin) obj.gy(mskin)];            
            xq = xgrid;
            Pwmq = obj.atlas.Pwm(mskin);
            Pgmq = obj.atlas.Pgm(mskin);
            phiq = obj.atlas.phi(mskin);
            uq = obj.fdmsol.uend(mskin);
            tq = obj.tend * ones(size(uq));

        end
       
        function [uq, xq, tq, Pwmq, Pgmq, phiq] = genGridDataUall(obj,mskin)
            % chose all time data from grid
            assert(obj.xdim == 2,'only in 2d');
            
            xgrid = [obj.gx(mskin) obj.gy(mskin)];
            
            % repeat ti in space, then reshape to 1d
            tmp = repmat(obj.fdmsol.tall,size(xgrid,1),1);
            tq = tmp(:);

            tlen = length(obj.fdmsol.tall);
            repeat = @(x) repmat(x, tlen, 1);

            
            xq = repeat(xgrid);

            Pwmq = repeat(obj.atlas.Pwm(mskin));
            Pgmq = repeat(obj.atlas.Pgm(mskin));
            phiq = repeat(obj.atlas.phi(mskin));

            u = zeros(length(phiq),1);
            uq = [];
            for i = 1:tlen
                uslice = obj.fdmsol.uall(:,:,i);
                uq = [uq;uslice(mskin)];
            end

            p = randperm(length(uq));
            [uq, xq, tq, Pwmq, Pgmq, phiq] = mapfun(@(x) x(p,:),uq, xq, tq, Pwmq, Pgmq, phiq);

        end
        
        function X = transformDat(obj, x, t)
            % center, scale, combine 
            X = [t/obj.tend (x - obj.x0)/obj.L];
        end




        function dat = genScatterData(obj, varargin)
            % generate scatter data set
            % (1) sample xq and tq, if finalt, set tq = 1
            % (2) based on (xq,tq), interpolate Pwm, Pm, phi, uend. might add noise
            
            p.n = 10000;
            p.finalt = false;
            p.usegrid = false; % use grid data
            p.method = 'linear';
            p.alltime = false;
            p.usepolar = false;
            p = parseargs(p, varargin{:});
            
            if p.usegrid
                % use grid data
                [~, ~, mskin] = obj.getrmax();
                [uq, xq, tq, Pwmq, Pgmq, phiq] = genGridDataUend(obj,mskin);
                [uqend, tqend] = deal(uq, tq);
                return
            end

            % generate x sample in circle, then interpolate
            xq = obj.xsample(varargin{:});

            tqend = ones(p.n,1)*obj.tend; % all final time
            tq = rand(p.n,1)*obj.tend; % sample t, unit
            
            
            if p.usepolar
                fprintf('interpolate polar solution');
                rq = x2r(xq);
                xgrid  = obj.polarsol.xgrid;
                tgrid  = reshape(obj.polarsol.tgrid,[],1);
                phiq = interp1(xgrid, obj.polarsol.phi, rq );
                uqend = interp1(xgrid, obj.polarsol.sol(end,:), rq );
                uqnn = uqend;
                Pwmq = obj.polargeo.Pwm(rq);
                Pgmq = obj.polargeo.Pgm(rq);

                uq = interpn(tgrid, xgrid, obj.polarsol.sol, tq, rq, p.method); % u(tq, xq), % for testing
                return
            end
            
            
            dat.uq = obj.interpu(tq, xq, p.method); % u(tq, xq), mainly for testing

            % useful for xdat and xtest
            dat.uqnn = obj.interpf(obj.fdmsol.uend, xq, p.method); % data no noise
            dat.nzuend = addNoise(obj.fdmsol.uend, varargin{:});
            dat.phiq = obj.interpf(obj.atlas.phi, xq, p.method);
            dat.uqend = obj.interpf(nzuend, xq, p.method);

            % Pwmq and Pgmq only useful of xr
            % interpolate anyway, might not be useful for testing data
            dat.Pwmq = obj.interpf(obj.atlas.Pwm, xq, p.method);
            dat.Pgmq = obj.interpf(obj.atlas.Pgm, xq, p.method);
            dat.DxDphi = obj.interpf(obj.fdmsol.DxDphi, xq, p.method);
            dat.DyDphi = obj.interpf(obj.fdmsol.DyDphi, xq, p.method);
            dat.DzDphi = obj.interpf(obj.fdmsol.DzDphi, xq, p.method);

        end

        function [fp,dataset] = scatterSample(obj, xrArg, xdatArg, xtestArg, varargin)
            % sample scatter for xr, 
            % sample xdat on grid,
            p.savedat = true;
            p.seed = 1;
            p.tag = 'sample';
            p.samex = true;  % xdat same as xr
            p.datdir = './';
            p = parseargs(p, varargin{:});
            seed = p.seed;
            rng(seed,'twister');
            
            % sample xr scattered
%             [Pwmq,Pgmq,phiq,uq,xq,tq, uqend, udatnn, tqend] = obj.genScatterData(xrArg{:});
            Xdat = obj.genScatterData(xdatArg{:});

            
            if p.samex
                fprintf('space xr same as xdat\n');
                [Pwmq, Pgmq, phiq, xq] = deal(Pwmdat, Pgmdat, phidat, xdat);
            else
                fprintf('space xr different from xdat\n');
                [Pwmq,Pgmq,phiq,uq,xq,tq, ~, ~, ~] = obj.genScatterData(xrArg{:});
            end
            
            xdat = obj.transformDat(xdat, tdat);
            xr = obj.transformDat(xq, tq);

            [~,~,phitest,utest,xtest,ttest,~,~] = obj.genScatterData(xtestArg{:});
            xtest = obj.transformDat(xtest, ttest);
            

            dataset = TrainDataSet;

            dataset.addvar(xdat,udat,phidat,udatnn,...
            xtest,utest,phitest,...
            Pwmq, Pgmq, phiq, xr, uq, seed,...
            xrArg, xdatArg, xtestArg);

            dataset.copyprop(obj, 'dwc','rhoc','L','T','DW','RHO','rDe','rRHOe',...
            'dw','dg','rho','x0','ix','xdim','tend','zslice');

            fp = sprintf('dat_%s_%s.mat',obj.name,p.tag);
            fp = fullfile(p.datdir, fp);

            if p.savedat
                dataset.save(fp);
                fprintf('save training dat to %s\n', fp );
            end

        end

        function fp = genGridData(obj,varargin)

            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'seed', 1);
            addParameter(p, 'tag', '');
            addParameter(p, 'softic', false);
            addParameter(p, 'uthresh', 0);
            addParameter(p, 'datdir', './'); % which directory to save
            parse(p, varargin{:});
            datdir = p.Results.datdir;
            uthresh = p.Results.uthresh;

            tag = p.Results.tag;
            if ~isempty(tag)
                tag = "_"+tag;
            end
            seed = p.Results.seed;
            rng(seed,'twister');

            [~, ~, mskin] = obj.getrmax();

            % sample residual points
            [uq, xq, tq, Pwmq, Pgmq, phiq] = obj.genGridDataUall(mskin);
            xr = obj.transformDat(xq, tq);

            % only pick u greater than threshold for residual points
            if uthresh > 0
                f = @(x) x(uq>uthresh,:);
                [uq, xr, Pwmq, Pgmq, phiq] = mapfun(f, uq, xr, Pwmq, Pgmq, phiq);
            end

            % test data
            xtest = xr;
            utest = uq;
            phitest = phiq;

            % sample data points
            if p.Results.softic == true
                ts = [1, length(obj.fdmsol.tall)];
            else
                ts = [length(obj.fdmsol.tall)];
            end
            [udat, xdat, tdat, phidat] = obj.genGridDataUt(mskin, ts);
            xdat = obj.transformDat(xdat, tdat);


            dataset = DataSet(xdat,udat,phidat,...
            xtest,utest,phitest,...
            Pwmq, Pgmq, phiq,xr, seed);
            dataset.copyprop(obj, 'dwc','rhoc','L','T','DW','RHO','rDe','rRHOe',...
            'dw','dg','rho','x0','ix','xdim','tend','zslice');
            
            if p.Results.savedat
                dataset.save(fp);
                fprintf('save training dat to %s\n', fp );
            end
        end


        function [xq] = xsample(obj,varargin)
            % sample spatial points
            p.n = 10000;
            p.uniformx = false;
            p.wgrad = false;
            p.nwratio = 0.0;
            p.radius = obj.rmax;
            p = parseargs(p,varargin{:});

            n_basic = p.n * (1-p.nwratio);
            n_enhance = p.n * (p.nwratio);

            xq = sampleDenseBall(n_basic, obj.xdim, p.radius, obj.x0, p.uniformx); 
            fprintf('dense sample %g, radius %g, uniformx = %g\n', n_basic, p.radius, p.uniformx);

            if p.wgrad == true
                ntmp = n_enhance*10;
                fprintf('weighted sample by graddf %g\n', n_enhance);
                
                xtmp = sampleDenseBall(ntmp, obj.xdim, p.radius, obj.x0, true); % uniform temporary 
                [dxdf, dydf] = gradient(obj.atlas.df.*obj.atlas.phi);
                normgrad = dxdf.^2 + dydf.^2;
                normgrad = imgaussfilt(normgrad, 1,'FilterDomain','spatial');
                normgradq = obj.interpf(normgrad, xtmp, 'linear');

                idx = datasample(1:ntmp, n_enhance, 'Replace',false, 'Weights', normgradq);
                f = @(x) x(idx,:);
                [xtmp] = mapfun(f, xtmp);

                xq = [xq; xtmp];
                xq = xq(randperm(size(xq, 1)), :);
            end
        end

        function [fp,dataset] = ReadyDat(obj, n, argsample, varargin)
            % prepare data for training
            p = inputParser;
            p.KeepUnmatched = true;
            addParameter(p, 'noiseon', 'uqe'); %none = no noise, uq noise after interp, u interp after noise
            addParameter(p, 'method', 'linear');
            addParameter(p, 'savedat', true);
            addParameter(p, 'isuniform', false);
            addParameter(p, 'finalxr', false); % final time as res pts
            addParameter(p, 'wsample', 0); % weight downsample
            addParameter(p, 'wmethod', 'u'); % how to weight sample
            addParameter(p, 'tweight', false); % weighted by time
            addParameter(p, 'seed', 1);
            addParameter(p, 'tag', '');
            addParameter(p, 'datdir', './'); % which directory to save
            parse(p, varargin{:});
            
            seed = p.Results.seed;
            datdir = p.Results.datdir;
            isuniform = p.Results.isuniform;

            [status,msg,msgid] = mkdir(datdir);
            assert(status);
            method = p.Results.method;
            tag = p.Results.tag;
            if ~isempty(tag)
                tag = "_"+tag;
            end

            % use data from interpolation
            rng(seed,'twister');
            % sample coord, with unit, radius is rmax,
            xq = xsample(obj,n, argsample{:});
            tq = rand(n,1)*obj.tend; % sample t, unit

            if p.Results.tweight == true
                fprintf('weighted t sample %g\n',n);
                tq = datasample(linspace(0,obj.tend,2*n)', n, 'Replace',false, 'Weights', linspace(0,1,2*n));
            end

            Pwmq = obj.interpf(obj.atlas.Pwm, xq, method);
            Pgmq = obj.interpf(obj.atlas.Pgm, xq, method);
            phiq = obj.interpf(obj.atlas.phi, xq, method);
            uq = obj.interpu(tq, xq, method); % u(tq, xq)
            
            % final time data
            xqe = xq;
            tqe = ones(n,1)*obj.tend; % all final time
            uqe = obj.interpf(obj.fdmsol.uend, xq, method); % u(t_end,xq), without noise

            % down sample
            wn = p.Results.wsample;
            if wn>0 && startsWith(p.Results.wmethod,'u');
                fprintf('weighted subsuample %g\n',wn);
                if strcmp(p.Results.wmethod,'u')
                    fprintf('weighted by u\n');
                    uw = uq;
                    uwe = uqe;
                elseif strcmp(p.Results.wmethod,'u2')
                    fprintf('weighted by u(1-u)\n');
                    uw = uq.*(1-uq);
                    uwe = uqe.*(1-uqe);
                else
                    error('unknow method');
                end
                
                % 
                idx = datasample(1:length(uq), wn, 'Replace',false, 'Weights', uw);
                f = @(x) x(idx,:);
                [uq, tq, xq, Pwmq, Pgmq, phiq] = mapfun(f, uq, tq, xq, Pwmq, Pgmq, phiq);
                
                
                idx_end = datasample(1:length(uqe), wn, 'Replace',false, 'Weights', uwe);
                f = @(x) x(idx_end,:);
                [uqe, xqe, tqe] = mapfun(f, uqe, xqe, tqe);
            end


            



            if strcmpi(p.Results.noiseon, 'uqe')
                % interp then add noise
                fprintf('interp then add noise\n');
                nzuqe = addNoise(uqe, p.Unmatched); % noisy uqe
            elseif strcmpi(p.Results.noiseon, 'uend')
                % add noise to uend then interp
                fprintf('add noise then interp\n');
                nzuend = addNoise(obj.fdmsol.uend, p.Unmatched);
                nzuqe = obj.interpf(nzuend, xq, method); % u(t_end,xq)
            else
               error('unknown option') 
            end

            % single point data 
            xdat = obj.transformDat(xqe, tqe);
            udat = nzuqe;
            phidat = phiq;

            % testing data, all time
            xr = obj.transformDat(xq, tq);
            xtest = xr;
            utest = uq;
            phitest = phiq;

            if p.Results.finalxr == true
                fprintf('eval res also at xdat')
                xr = [xr; xdat];
                Pwmq = [Pwmq; Pwmq];
                Pgmq = [Pgmq; Pgmq];
                phiq = [phiq; phiq];
            end
            
            fp = sprintf('dat_%s_n%d%s.mat',obj.name,n,tag);
            fp = fullfile(datdir, fp);

            dataset = DataSet(xdat,udat,phidat,...
                    xtest,utest,phitest,...
                    Pwmq, Pgmq, phiq,xr, seed);
            dataset.copyprop(obj, 'dwc','rhoc','L','T','DW','RHO','rDe','rRHOe',...
            'dw','dg','rho','x0','ix','xdim','tend','zslice');
            if p.Results.savedat
                dataset.save(fp);
                fprintf('save training dat to %s\n', fp );
            end
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

        function plotline(obj, dat,name)
            % 1d plot of data, mid section
            
%             [~,mid] = getBox(dat, 0.01, 0);
%             y = dat(:,obj.x0(2));
            y = dat(obj.x0(1),:);
            x = 1:length(y);
            plot(x,y,'DisplayName',name);
            
        end
    end
end
