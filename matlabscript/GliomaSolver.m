classdef GliomaSolver< handle
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
        phi
        uall
        tall
        uend
        
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
            obj.xdim = xdim;
            obj.dw = d;
            obj.dg = d/10;
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
            if length(varargin)==1 &&isa(varargin{1},'Atlas')
                obj.atlas = varargin{1};
            else
                obj.atlas = Atlas('dw', obj.dw, 'zslice',obj.zslice, varargin{:});
            end
            sz = [1 1 1];
            sz(1:obj.xdim) = size(obj.atlas.Pwm); % even when Pwm is 2d, 3rd dim is 1
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
            assert(isfile(fp));

            dat = load(fp);
            obj.uall = dat.uall;
            obj.tall = dat.tall;
            obj.phi = dat.phi;
            if obj.xdim == 2
                obj.uend = obj.uall(:,:,end);
            else
                obj.uend = obj.uall(:,:,:,end);
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
            p = parseargs(p, varargin{:});


            [fp,es] = obj.solpath;
            if es && p.redo == false
                fprintf('load existing solution\n');
                obj.loadsol(fp);
            else
                [obj.phi,obj.uall,obj.tall,obj.uend] = GliomaFdmSolve(obj.atlas.Pwm, obj.atlas.Pgm, obj.atlas.Pcsf, obj.dw, obj.rho, obj.tend, obj.ix, obj.xdim);
                if p.savesol
                    fprintf('save pde solution\n');
                    obj.savesol(fp); 
                end
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
            uend = slice2d(obj.uend,varargin{:});
            [ax1, ax2]  = obj.atlas.imagesc2('df', uend, varargin{:});
        end

        function [fig, ax1, ax2] = plotphiuend(obj,varargin)
            fig = figure;
            uend = slice2d(obj.uend,varargin{:});
            phi = slice2d(obj.phi,varargin{:});
            [ax1, ax2]  = obj.atlas.imagesc2('df', uend.*phi, varargin{:});
        end

        function [rmax, idxin, mskin] = getrmax(obj)
            % get radius of solution
            padding = 1; % padding ot max distance
            threshold = 0.01; % threshold for tumor region

            tk = length(obj.tall); 
            idx = find(obj.uend.*obj.phi>threshold); % index of u value greater than threshold
            
            distix = sqrt((obj.gx-obj.ix(1)).^2+ (obj.gy-obj.ix(2)).^2+(obj.gz-obj.ix(3)).^2);
            [maxdis,maxidx] = max(distix(idx));
            rmax  = maxdis + padding;
            obj.rmax = rmax;

            lsf = distix - obj.rmax; % level set function
            mskin = lsf<0;
            idxin = find(lsf<0);
            fprintf('rmax = %g\n',obj.rmax);
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
                fq = interpn(xg, yg, zg, obj.tall, obj.uall, X(:,1), X(:,2), X(:,3), t, method);
            else
                xg = f(obj.gx(:,1));
                yg = f(obj.gy(1,:));
                fq = interpn(xg, yg, obj.tall, obj.uall, X(:,1), X(:,2), t, method);
            end 
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
        end
       
        function [uq, xq, tq, phiq] = genGridDataUt(obj,mskin,ts)
            % generate final time data from grid,
            if nargin < 3
                ts = length(obj.tall);
            end
            
            assert(obj.xdim == 2,'only in 2d');
            
            xq = [];
            phiq = [];
            uq = [];
            tq = [];
            for i = 1:length(ts)
                tk = ts(i);
                fprintf('sample grid data at t = %g\n',obj.tall(tk));
                xq = [xq;[obj.gx(mskin) obj.gy(mskin)]];
                phiq = [phiq;obj.phi(mskin)];
                u = obj.uall(:,:,tk);
                uq = [uq; u(mskin)];
                tq = [tq; ones(length(uq),1) * obj.tall(tk)];
            end

            p = randperm(length(uq));
            [uq, xq, tq, phiq] = mapfun(@(x) x(p,:),uq, xq, tq, phiq);

        end


       
        function [uq, xq, tq, Pwmq, Pgmq, phiq] = genGridDataUall(obj,mskin)
            % chose all time data from grid
            assert(obj.xdim == 2,'only in 2d');
            
            xgrid = [obj.gx(mskin) obj.gy(mskin)];
            
            % repeat ti in space, then reshape to 1d
            tmp = repmat(obj.tall,size(xgrid,1),1);
            tq = tmp(:);

            tlen = length(obj.tall);
            repeat = @(x) repmat(x, tlen, 1);

            
            xq = repeat(xgrid);

            Pwmq = repeat(obj.atlas.Pwm(mskin));
            Pgmq = repeat(obj.atlas.Pgm(mskin));
            phiq = repeat(obj.atlas.phi(mskin));

            u = zeros(length(phiq),1);
            uq = [];
            for i = 1:tlen
                uslice = obj.uall(:,:,i);
                uq = [uq;uslice(mskin)];
            end

            p = randperm(length(uq));
            [uq, xq, tq, Pwmq, Pgmq, phiq] = mapfun(@(x) x(p,:),uq, xq, tq, Pwmq, Pgmq, phiq);

        end
        
        function X = transformDat(obj, x, t)
            % center, scale, combine 
            X = [t/obj.tend (x - obj.x0)/obj.L];
        end


        function [fp,dataset] = scattergrid(obj, xrArg, varargin)
            % sample scatter for xr, 
            % sample xdat on grid,
            p.savedat = true;
            p.seed = 1;
            p.tag = 'scattergrid';
            p.datdir = './';
            p = parseargs(p, varargin{:});

            seed = p.seed;
            rng(seed,'twister');

            % sample xr scattered
            xq = obj.xsample(xrArg{:});
            tq = rand(length(xq),1)*obj.tend; % sample t, unit
            xr = obj.transformDat(xq, tq);

            method = 'linear';
            Pwmq = obj.interpf(obj.atlas.Pwm, xq, method);
            Pgmq = obj.interpf(obj.atlas.Pgm, xq, method);
            phiq = obj.interpf(obj.phi, xq, method);
            uq = obj.interpu(tq, xq, method); % u(tq, xq)

            % sample xdat from grid
            [~, ~, mskin] = obj.getrmax();
            [udat, xdat, tdat, phidat] = obj.genGridDataUt(mskin, length(obj.tall));
            xdat = obj.transformDat(xdat, tdat);

            % test data
            xtest = xr;
            utest = uq;
            phitest = phiq;
            
            dataset = TrainDataSet;

            dataset.addvar(xdat,udat,phidat,...
            xtest,utest,phitest,...
            Pwmq, Pgmq, phiq,xr, seed);
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
                ts = [1, length(obj.tall)];
            else
                ts = [length(obj.tall)];
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


        function [xq] = xsample(obj, n, varargin)
            p.isuniform = false;
            p.wgrad = false;
            p.nwratio = 0.0;
            p = parseargs(p,varargin{:});

            n_basic = n * (1-p.nwratio);
            n_enhance = n * (p.nwratio);

            xq = sampleDenseBall(n_basic, obj.xdim, obj.rmax, obj.x0, p.isuniform); 
            fprintf('dense sample  %g\n', n_basic);

            if p.wgrad == true
                ntmp = n_enhance*10;
                fprintf('weighted sample by graddf %g\n', n_enhance);
                
                xtmp = sampleDenseBall(ntmp, obj.xdim, obj.rmax, obj.x0, true); % uniform temporary 
                [dxdf, dydf] = gradient(obj.atlas.df.*obj.phi);
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
            phiq = obj.interpf(obj.phi, xq, method);
            uq = obj.interpu(tq, xq, method); % u(tq, xq)
            
            % final time data
            xqe = xq;
            tqe = ones(n,1)*obj.tend; % all final time
            uqe = obj.interpf(obj.uend, xq, method); % u(t_end,xq), without noise

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
                nzuend = addNoise(obj.uend, p.Unmatched);
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



