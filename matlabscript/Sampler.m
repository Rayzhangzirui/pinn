classdef Sampler< dynamicprops
    properties
        setting
        model
        
        % spatial sample of xr, xtest, xdat, used to generate training data
        sxr
        sxtest
        sxdat
        sxbc

        datlres % data for residual loss, P, phi, DxyzPphi
        datldat % data for data loss
        datlbc % data for soft bc loss
        datltest % data for testing loss
        dateval % data for evaluating
    end

    methods
        function obj = Sampler(model,varargin)
            obj.model = model;

            obj.setting.interp_method = 'linear';
            obj.setting.samex = true;
            obj.setting.savedat = true;
            obj.setting.seed = 1;
            obj.setting.tag = 'sample';
            obj.setting.datdir = './';
            obj.setting.radius = model.rmax+2;
            obj.setting.usepolar = false;
            obj.setting.urange = [-inf inf];
            obj.setting.uth = [];

            
        
            obj.setting = parseargs(obj.setting, varargin{:});
            rng(obj.setting.seed,'twister');

            obj.setting.noise.type = 'none';
            obj.setting.noise.mu = 0;
            obj.setting.noise.std = 0.1;
            obj.setting.noise.fsig = 3;
            obj.setting.noise.threshold = [0,1];

        end

        function setnoise(obj,varargin)
            obj.setting.noise = parseargs(obj.setting.noise, varargin{:});
        end

        function setxsample(obj,datname,varargin)
            p.n = 20000;
            p.isuniformx = false; % uniform distribution or not
            p.issphere = false; 
            p.radius = obj.setting.radius;
            p.x0 = obj.model.x0;
            p.nwratio = 0.0; % ratio of data that are weighted
            p.wdata = [];  % weighted by some data
            p.sameas = '';
            p = parseargs(p, varargin{:});
            obj.setting.(datname) = p;
        end


        function xq = xsample(obj, datname)

            p = obj.setting.(datname);

            if ~isempty(p.sameas)
                xq = obj.(p.sameas);
                fprintf('datname same as %s\n',p.sameas);
                return
            end

            n_basic = p.n * (1-p.nwratio);
            n_enhance = p.n * (p.nwratio);

            xq = sampleDenseBall(n_basic, obj.model.xdim, p); 
            fprintf('%s: sample %g, radius %g, uniformx = %g\n', datname, n_basic, p.radius, p.isuniformx);

            if ~isempty(p.wdata)
                ntmp = n_enhance*10;
                fprintf('weighted sample');
                
                xtmp = sampleDenseBall(ntmp, obj.model.xdim, radius, obj.model.x0, true); % uniform temporary 
                % [dxdf, dydf] = gradient(p.wdata);
                % normgrad = dxdf.^2 + dydf.^2;
                % normgrad = imgaussfilt(normgrad, 1,'FilterDomain','spatial');
                weight = obj.model.interpf(p.wdata, xtmp, obj.setting.interp_method);

                idx = datasample(1:ntmp, n_enhance, 'Replace',false, 'Weights', weight);
                f = @(x) x(idx,:);
                [xtmp] = mapfun(f, xtmp);

                xq = [xq; xtmp];
                xq = xq(randperm(size(xq, 1)), :);
            end 

        end



        function dat = genDatLdat(obj,xq)
            % interpolate fdm solution
            method = obj.setting.interp_method;
            ndat = size(xq,1);
            tqend = ones(ndat, 1) * obj.model.tend; % all final time
        
            noise_uend = addNoise(obj.model.fdmsol.uend, obj.setting.noise);

            % useful for xdat and xtest

            if obj.setting.usepolar
                rq = x2r(xq);
                xgrid  = obj.model.polarsol.xgrid;
                dat.udat = interp1(xgrid, obj.model.polarsol.sol(end,:), rq, method );
                dat.phidat = obj.model.polargeo.phi(rq);
            else
                dat.udatnn = obj.model.interpf(obj.model.fdmsol.uend, xq, method); % data no noise
                dat.udat = obj.model.interpf(noise_uend, xq, method); % data might with noise
                dat.phidat = obj.model.interpf(obj.model.atlas.phi, xq, method);
            end
            
            
            dat.xdat = obj.model.transformDat(xq, tqend);
            
            % data for evaluating nn
            obj.dateval.xeval = dat.xdat;
            obj.dateval.phieval = dat.phidat;
            obj.dateval.ueval = dat.udatnn;


            % down sample every
            if isempty(obj.setting.uth)
                idx = dat.udat>obj.setting.urange(1) & dat.udat<obj.setting.urange(2);
                dat = structfun(@(x) x(idx,:), dat, 'UniformOutput', false);
                fprintf('urange down sample %d to %d \n',ndat, sum(idx));
            else
                % threshold udat to [0, uth(1), uth(2)]
                top = dat.udat>obj.setting.urange(2);
                top2 = dat.udat>obj.setting.urange(1);
                dat.udat(top2) = obj.setting.uth(1);
                dat.udat(top) = obj.setting.uth(2);
                dat.udat(~top2) = 0.0;
                fprintf('threshold to [0, %g, %g ]\n', obj.setting.uth);
            end

            dat.plfdat = 4*dat.udat.*(1 - dat.udat); % proliferation, scaled to [0,1]
            
            
            
            obj.datldat= dat;
        end

        

        function dat = genDatLres(obj,xq)
            % data for residual

            method = obj.setting.interp_method;
            nxr = size(xq,1);
            tq = rand(nxr,1)*obj.model.tend; % sample t, unit

            if obj.setting.usepolar
                rq = x2r(xq);
                
                dat.phiq = obj.model.polargeo.phi(rq);
                dat.Pq = obj.model.polargeo.P(rq);
                [dat.DxPphi, dat.DyPphi] = obj.model.evalVectorPolar(obj.model.polargeo.DxPphi, xq(:,1),xq(:,2));
                dat.DzPphi = zeros(size(dat.DxPphi));
                
            else
                dat.phiq = obj.model.interpf(obj.model.atlas.phi, xq, method);
                dat.Pq = obj.model.interpf(obj.model.atlas.P, xq, method);
                dat.DxPphi = obj.model.interpf(obj.model.fdmsol.DxPphi, xq, method);
                dat.DyPphi = obj.model.interpf(obj.model.fdmsol.DyPphi, xq, method);
                dat.DzPphi = obj.model.interpf(obj.model.fdmsol.DzPphi, xq, method);
            end

            dat.xr = obj.model.transformDat(xq, tq);

            obj.datlres= dat;
        end

        function dat = genDatTest(obj,xq)
            % data for testing

            method = obj.setting.interp_method;
            ntest = size(xq,1);
            
            tq = rand(ntest,1)*obj.model.tend; % sample t, unit

            dat.utest = obj.model.interpu(tq, xq, method); % u(tq, xq), mainly for testing
            dat.phitest = obj.model.interpf(obj.model.atlas.phi, xq, method);

            dat.xtest = obj.model.transformDat(xq, tq);
    
            obj.datltest= dat;
        end

        function dat = genDatBc(obj,xq)
            % data for testing

            method = obj.setting.interp_method;
            n = size(xq,1);

            tq = rand(n,1)*obj.model.tend; % sample t, unit
            dat.ubc = obj.model.interpu(tq, xq, method); % u(tq, xq), mainly for testing
            dat.phibc = obj.model.interpf(obj.model.atlas.phi, xq, method);
            dat.xbc = obj.model.transformDat(xq, tq);
            obj.datlbc= dat;
        end

        

        function genScatterData(obj)
            obj.sxdat = obj.xsample('xdatopt');
            obj.genDatLdat(obj.sxdat);
            
            obj.sxr = obj.xsample('xropt');
            obj.genDatLres(obj.sxr);

            obj.sxtest = obj.xsample('xtestopt');
            obj.genDatTest(obj.sxtest);
            
            if isfield(obj.setting,'xbcopt')
                obj.sxbc = obj.xsample('xbcopt');
                obj.genDatBc(obj.sxbc);
            end
        end


        function dataset = genDataSet(obj)
            % generate data set for training

            dataset = TrainDataSet();

            dataset.copyprop(obj.model, 'dwc','rhoc','L','T','DW','RHO','rDe','rRHOe',...
            'dw','dg','rho','x0','ix','xdim','tend','zslice');

            dataset.copyprop(obj.datldat);
            dataset.copyprop(obj.datlres);
            dataset.copyprop(obj.datltest);
            dataset.copyprop(obj.setting);
            dataset.copyprop(obj.datlbc);

            fp = sprintf('dat_%s_%s.mat',obj.model.name,obj.setting.tag);
            fp = fullfile(obj.setting.datdir, fp);

            if obj.setting.savedat
                dataset.save(fp);
                fprintf('save training dat to %s\n', fp );
            end
        end

        


    end
end