classdef Sampler< dynamicprops
    properties
        setting
        model

        datlres % data for residual loss, P, phi, DxyzPphi
        datldat % data for data loss
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
            obj.setting.radius = model.rmax;
            obj.setting.usepolar = false;
            obj.setting.urange = [-inf inf];
            
        
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

        function obj = xsample(obj, datname, varargin)
            p.n = 20000; % total number of data point
            p.uniformx = false; % uniform distribution or not
            radius = obj.setting.radius;

            p.nwratio = 0.0; % ratio of data that are weighted
            p.wdata = [];  % weighted by some data
            p = parseargs(p,varargin{:});

            n_basic = p.n * (1-p.nwratio);
            n_enhance = p.n * (p.nwratio);

            xq = sampleDenseBall(n_basic, obj.model.xdim, radius, obj.model.x0, p.uniformx); 
            fprintf('sample %g, radius %g, uniformx = %g\n', n_basic, radius, p.uniformx);

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
            
            if ~isprop(obj,datname)
                addprop(obj, datname);
            end
            obj.(datname) = xq;

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
            idx = dat.udat>obj.setting.urange(1) & dat.udat<obj.setting.urange(2);
            dat = structfun(@(x) x(idx,:), dat, 'UniformOutput', false);
            fprintf('urange down sample %d to %d \n',ndat, sum(idx));
            
            obj.datldat= dat;
        end

        

        function dat = genDatLres(obj,xq)
            % data for residual
            if obj.setting.samex
                fprintf('xr same as xdat\n');
                xq = obj.xqdat;
            end

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

        

        function genScatterData(obj)
            obj.xsample('xqdat');
            obj.genDatLdat(obj.xqdat);
            obj.genDatLres();
            obj.xsample('xqtest');
            obj.genDatTest(obj.xqtest);
        end


        function dataset = genDataSet(obj)
            % generate data set for training

            dataset = TrainDataSet();

            dataset.copyprop(obj.model, 'dwc','rhoc','L','T','DW','RHO','rDe','rRHOe',...
            'dw','dg','rho','x0','ix','xdim','tend','zslice');

            dataset.copyprop(obj.datldat);
            dataset.copyprop(obj.datlres);
            dataset.copyprop(obj.datltest);
            dataset.copyprop(obj.setting.noise);
            dataset.copyprop(obj.dateval);

            fp = sprintf('dat_%s_%s.mat',obj.model.name,obj.setting.tag);
            fp = fullfile(obj.setting.datdir, fp);

            if obj.setting.savedat
                dataset.save(fp);
                fprintf('save training dat to %s\n', fp );
            end
        end

        


    end
end