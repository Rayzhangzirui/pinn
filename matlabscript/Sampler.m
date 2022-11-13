classdef Sampler< dynamicprops
    properties
        setting
        model
    end

    methods
        function obj = Sampler(model,varargin)
            obj.setting.interp_method = 'linear';
            obj.setting.samex = true;
            obj.setting.savedat = true;
            obj.setting.seed = 1;
            obj.setting.tag = 'sample';
            obj.setting.datdir = './';
            
        
            obj.setting = parseargs(obj.setting, varargin{:});
            rng(obj.setting.seed,'twister');


            obj.setting.noise.type = 'none';
            obj.model = model;
            
        end

        function setnoise(obj,varargin)
            obj.setting.noise = parseargs(obj.setting.noise, varargin{:});
        end

        function obj = xsample(obj, datname, varargin)
            p.n = 20000; % total number of data point
            p.uniformx = false; % uniform distribution or not
            p.radius = obj.model.rmax;

            p.nwratio = 0.0; % ratio of data that are weighted
            p.wdata = [];  % weighted by some data
            p = parseargs(p,varargin{:});

            n_basic = p.n * (1-p.nwratio);
            n_enhance = p.n * (p.nwratio);

            xq = sampleDenseBall(n_basic, obj.model.xdim, p.radius, obj.model.x0, p.uniformx); 
            fprintf('sample %g, radius %g, uniformx = %g\n', n_basic, p.radius, p.uniformx);

            if ~isempty(p.wdata)
                ntmp = n_enhance*10;
                fprintf('weighted sample');
                
                xtmp = sampleDenseBall(ntmp, obj.model.xdim, p.radius, obj.model.x0, true); % uniform temporary 
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
            dat.ndat = size(xq,1);
            tqend = ones(dat.ndat, 1) * obj.model.tend; % all final time
        
            noise_uend = addNoise(obj.model.fdmsol.uend, obj.setting.noise);

            % useful for xdat and xtest
            dat.udatnn = obj.model.interpf(obj.model.fdmsol.uend, xq, method); % data no noise
            dat.udat = obj.model.interpf(noise_uend, xq, method); % data might with noise
            dat.phidat = obj.model.interpf(obj.model.atlas.phi, xq, method);
            
            dat.xdat = obj.model.transformDat(xq, tqend);
            dat.xq = xq;
            
            if ~isprop(obj,'datldat')
                addprop(obj, 'datldat');
            end
            obj.datldat= dat;
        end


        function dat = genDatLres(obj,xq)
            % data for residual
            if obj.setting.samex
                fprintf('xr same as xdat\n');
                xq = obj.datldat.xq;
            end

            method = obj.setting.interp_method;
            dat.nxr = size(xq,1);
            tq = rand(dat.nxr,1)*obj.model.tend; % sample t, unit

            % useful for xdat and xtest
            dat.phiq = obj.model.interpf(obj.model.atlas.phi, xq, method);
            dat.Pwmq = obj.model.interpf(obj.model.atlas.Pwm, xq, method);
            dat.Pgmq = obj.model.interpf(obj.model.atlas.Pgm, xq, method);
            dat.DxDphi = obj.model.interpf(obj.model.fdmsol.DxDphi, xq, method)/obj.model.dw;
            dat.DyDphi = obj.model.interpf(obj.model.fdmsol.DyDphi, xq, method)/obj.model.dw;
            dat.DzDphi = obj.model.interpf(obj.model.fdmsol.DzDphi, xq, method)/obj.model.dw;

            dat.xr = obj.model.transformDat(xq, tq);

            if ~isprop(obj,'datlres')
                addprop(obj, 'datlres');
            end
            obj.datlres= dat;
        end

        function dat = genDatTest(obj,xq)
            % data for testing

            method = obj.setting.interp_method;
            dat.ntest = size(xq,1);
            
            tq = rand(dat.ntest,1)*obj.model.tend; % sample t, unit

            dat.utest = obj.model.interpu(tq, xq, method); % u(tq, xq), mainly for testing
            dat.phitest = obj.model.interpf(obj.model.atlas.phi, xq, method);
            dat.xtest = obj.model.transformDat(xq, tq);
    

            if ~isprop(obj,'datltest')
                addprop(obj, 'datltest');
            end
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
            dataset = TrainDataSet();

            dataset.copyprop(obj.model, 'dwc','rhoc','L','T','DW','RHO','rDe','rRHOe',...
            'dw','dg','rho','x0','ix','xdim','tend','zslice');

            dataset.copyprop(obj.datldat);
            dataset.copyprop(obj.datlres);
            dataset.copyprop(obj.datltest);
            
            fp = sprintf('dat_%s_%s.mat',obj.model.name,obj.setting.tag);
            fp = fullfile(obj.setting.datdir, fp);

            if obj.setting.savedat
                dataset.save(fp);
                fprintf('save training dat to %s\n', fp );
            end
        end

        


    end
end