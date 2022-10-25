classdef PostData<dynamicprops
   % class to handle postprocessing data
   properties
	  modeldir
	  tag
	  udat  %
	  upred % predictioin at residual points
	  log % table of training log
	  info  % information of model
	  trainDataSet % info or training data
	  history
	  savefig
	  atlas
	  fwdmodel
	  fwdmodele
   end
   
   methods
	  function obj = PostData(modeldir,varargin)
		  p = inputParser;
		  addRequired(p,'modeldir',@ischar);
		  addParameter(p,'savefig',true,@isbool);
		  addParameter(p,'udat','',@ischar);
		  addParameter(p,'tag','',@ischar);
		  addParameter(p,'dim',2,@isscalar);
		  addParameter(p,'isradial',false,@islogical);
		  addParameter(p,'dat_file','',@ischar);
		  parse(p,modeldir,varargin{:});
		  obj.savefig = p.Results.savefig;
		  obj.modeldir = p.Results.modeldir;
		  
		  assert(exist(obj.modeldir, 'dir')==7,'dir does not exist');
		  obj.tag = p.Results.tag;
		  fs = dir(obj.modeldir);          
		  
		  for i = 1:length(fs)
			  path = fullfile(fs(i).folder,fs(i).name);
			  
			  if contains(fs(i).name, 'history')
				  history = readtable(path);
				  
				  % down sampling
				  n = 10000;
				  N = size(history,1);
				  step = floor(N/n);
				  obj.history = history(1:step:end,:);
				  fprintf('read %s\n',path);
				  continue
			  end
			  
			  if contains(fs(i).name, 'solver.log')
				  [obj.log, ~] = readlog(path);
				  fprintf('read %s\n',path);
				  continue

			  end

			  if contains(fs(i).name,'json')
				  f = fileread(path);
				  obj.info = jsondecode(f);
				  fprintf('read %s\n',path);
				  continue
			  end

			  if startsWith(fs(i).name,'upred') && endsWith(fs(i).name,'mat')
				dat = load(path);
				
				fn = fieldnames(dat);
				for k=1:numel(fn)
					if( isnumeric(dat.(fn{k})) )
						dat.(fn{k}) = double(dat.(fn{k}));
					end
				end

				obj.upred{end+1} = dat;
				obj.upred{end}.file = path;
			  end
		  end
			
		  % read data file
		  if ~isempty(obj.info)
			  fp = fullfile(obj.modeldir, obj.info.inv_dat_file);
			  found = false;
			  if isfile(fp)
				  found = true;
			  else
				  fp = fullfile(parentdir(obj.modeldir), obj.info.inv_dat_file);
				  if isfile(fp)
					  found = true;
				  end
			  end
			  if found
				  fprintf('read %s\n',fp);
				  obj.trainDataSet = DataSet(fp);
			  else
				  fprintf('data file not found %s\n',fp);
			  end
		  end

	  end %end of constructor

		function readmri(obj, varargin)
			obj.atlas = Atlas('dw', obj.trainDataSet.dw,'zslice',obj.trainDataSet.zslice);
		end

	   function [fig,sc] = PlotLoss(obj,varargin)
			fig = figure;
			sc(1) = plot(obj.log.it, log10(obj.log.total),'DisplayName', 'total',varargin{:});
			hold on;
			
			sc(2) = plot(obj.log.it, log10(obj.log.res),'DisplayName',   'res',varargin{:});
			sc(3) = plot(obj.log.it, log10(obj.log.data),'DisplayName', 'data',varargin{:} );
			sc(3) = plot(obj.log.it, log10(obj.log.tmse),'DisplayName', 'test',varargin{:} );
			grid on;
			xlabel('steps');
			ylabel('log10(loss)');
			legend('Location','best');
			title(obj.tag);
			export_fig(fullfile(obj.modeldir,'fig_loss.jpg'),'-m3');
	   end

	   function unscale(obj, xdim, T, L, x0)
		   % remove scaling of (t,x)
		   scale = [T ones(1,xdim)*L];
		   for i = 1:length(obj.upred)
			   if isfield(obj.upred{i},"xdat")
				   obj.upred{i}.xdat = obj.upred{i}.xdat .*scale + [0 x0];
			   end
			   obj.upred{i}.xr = obj.upred{i}.xr .*scale + [0 x0];
		   end
	   end

	   function [fig, sc] = PlotInferErr(obj, varargin)
		   % plot inference error
		   rDe = obj.trainDataSet.rDe;
		   rRHOe = obj.trainDataSet.rRHOe;
		   errD = relerr(rDe, obj.log.rD);
		   errRHO = relerr(rRHOe, obj.log.rRHO);

		   fig = figure;
		   ts = sprintf('%s rel err\n rD = %0.2e, rRHO = %0.2e\n',...
			   obj.tag, relerr(rDe,obj.upred{1}.rD),relerr(rRHOe,obj.upred{1}.rRHO))
			sc(1) = plot(obj.log.it, errD,'DisplayName', 'rD',varargin{:});
			hold on;
			sc(2) = plot(obj.log.it, errRHO,'DisplayName', 'rRHO',varargin{:} );
			grid on;
			title(ts);
			xlabel('steps');
			ylabel('rel err');
			legend('Location','best');
			
			fpath = fullfile(obj.modeldir,'fig_relerr.jpg');
			fprintf('rel err param saved to %s\n',fpath);
			export_fig(fpath,'-m3');
	   end

	   function scatterdist(obj)
		% plot distribution of xdat 
		   t = tiledlayout(1,2,'TileSpacing','compact');
			
		%    xdat = obj.upred{1}.xdat;
		%    udat = obj.upred{1}.udat;
		%    scatter3(xdat(:,2),xdat(:,3),xdat(:,1),6,udat)
		%    title('x_{dat}')

			xdat = obj.upred{1}.xr;
			utest = obj.upred{1}.utest;
			scatter3(xdat(:,2),xdat(:,3),xdat(:,1),6,utest)
			title('x_r')
		end


		function [ax1,ax2] = scatter(obj, X, data, tag)

			[ax1,ax2] = obj.atlas.scatter('df', X(:,2:end) , 8, data,...
			'filled','MarkerFaceAlpha',0.5);
			title(ax1,tag);
		end

		function scatterupred(obj)
            % prediction error
            ndat = obj.info.n_dat_pts;

            xdat = obj.upred{1}.xdat(1:ndat,:);
			upredxdat = obj.upred{1}.upredxdat;

			udat = obj.trainDataSet.udat(1:ndat);
			phidat = obj.trainDataSet.phidat(1:ndat);
            
            umax = min(max(udat),1);
                
			% plot pred u
			[ax1,ax2] = obj.scatter(xdat, upredxdat.*phidat, '\phi u_{pred}');
            clim(ax2, [0, umax]);
            
			fname = 'fig_upredxdat.jpg';
			if obj.savefig
				export_fig(fullfile(obj.modeldir,fname),'-m3');
			end

			% plot data u
			[ax1,ax2] = obj.scatter(xdat, udat.*phidat, '\phi u_{dat}');
            clim(ax2, [0, umax]);
			fname = 'fig_udat.jpg';
			if obj.savefig
				export_fig(fullfile(obj.modeldir,fname),'-m3');
			end

			% plot error u
			err = (upredxdat - udat ).*phidat;
			[ax1,ax2] = obj.scatter(xdat, err, '\phi|u_{pred}-u_{dat}|');

			fname = 'fig_uprederr.jpg';
			if obj.savefig
				export_fig(fullfile(obj.modeldir,fname),'-m3');
			end
		end

		function [ax1,ax2] = scatterres(obj)
            % scatter plot of residual 
            % note: not a good way of visualization, different t
			absres = abs(obj.upred{1}.res);
        
			nres = obj.info.n_res_pts;
			phi = obj.trainDataSet.phiq(1:nres);
            

            xr = obj.upred{1}.xr(1:nres,:);

			[ax1,ax2] = obj.scatter(xr, absres.*phi, '\phi|res|');
			fname = 'fig_res.jpg';
			if obj.savefig
				export_fig(fullfile(obj.modeldir,fname),'-m3');
			end
		end


		function forward(obj)
            % forward FDM solve using inferred parameter
			rD = obj.upred{1}.rD;
			rRHO = obj.upred{1}.rRHO;

			xim = obj.trainDataSet.xdim;

			dw2 = rD * obj.trainDataSet.DW * obj.trainDataSet.L^2;
			rho2 = rRHO * obj.trainDataSet.RHO;
			[xdim,x0,zslice,dw,rho,tend] = obj.trainDataSet.getvar('xdim','x0','zslice','dw','rho','tend');
			
			obj.fwdmodel = GliomaSolver(xdim,  dw2, rho2 ,x0, 1, zslice);
			obj.fwdmodel.readmri(obj.atlas);
			obj.fwdmodel.solve('redo',true,'savesol',false);

			obj.fwdmodele = GliomaSolver(xdim,  dw, rho ,x0, tend, zslice);
			obj.fwdmodele.readmri(obj.atlas);
			obj.fwdmodele.solve('savesol',false);
		end

		function fwderr(obj)
            % error of fwd solution using infered param
			[ax1,ax2] = obj.atlas.imagesc2('df', obj.fwdmodel.uend.*obj.fwdmodel.phi);
			title(ax1,'\phi u_{fdm,pred}');
			if obj.savefig
				export_fig(fullfile(obj.modeldir,'fig_fdm_upred.jpg'),'-m3')
			end
			

			err =  abs(obj.fwdmodel.uend - obj.fwdmodele.uend).*obj.fwdmodel.phi;
			[ax1,ax2] = obj.atlas.imagesc2('df', err);
			title(ax1,'\phi |u_{fdm,pred} - u_{end}|');
			if obj.savefig
				export_fig(fullfile(obj.modeldir,'fig_fdm_err.jpg'),'-m3')
			end

		end


   end
end