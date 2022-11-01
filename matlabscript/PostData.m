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
	  yessave
	  atlas
	  fwdmodel
	  fwdmodele
   end
   
   methods
	  function obj = PostData(modeldir,varargin)
		  p = inputParser;
		  addRequired(p,'modeldir',@ischar);
		  addParameter(p,'yessave',true,@islogical);
		  addParameter(p,'udat','',@ischar);
		  addParameter(p,'tag','',@ischar);
		  addParameter(p,'dim',2,@isscalar);
		  addParameter(p,'isradial',false,@islogical);
		  addParameter(p,'dat_file','',@ischar);
		  parse(p,modeldir,varargin{:});
		  obj.yessave = p.Results.yessave;
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
				obj.upred{end+1} = DataSet(path);
				obj.upred{end}.addvar(path);
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
			
            w_dat = obj.info.w_dat;
            datlos  = obj.log.data * w_dat;

			sc(2) = plot(obj.log.it, log10(obj.log.res),'DisplayName',   'res',varargin{:});
			sc(3) = plot(obj.log.it, log10(datlos),'DisplayName', [num2str(w_dat) 'data'],varargin{:} );
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
			   if isprop(obj.upred{i},"xdat")
				   obj.upred{i}.xdat = obj.upred{i}.xdat .*scale + [0 x0];
               end

               if isprop(obj.upred{i},"xr")
			        obj.upred{i}.xr = obj.upred{i}.xr .*scale + [0 x0];
               end
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
        
        function savefig(obj,fname,varargin)
            if obj.yessave
				fp = fullfile(obj.modeldir,fname);
				fprintf('save %s\n',fp);
				export_fig(fp,varargin{:});
            end
        end

		function scatterupred(obj)
            % prediction error

            % might inpt n_dat_pts> length of xdat by mistake
            % numpy just slice to the end
            ndat = min(size(obj.upred{1}.xdat,1),obj.info.n_dat_pts);

            xdat = obj.upred{1}.xdat(1:ndat,:);
			upredxdat = obj.upred{1}.upredxdat;

			udat = obj.trainDataSet.udat(1:ndat);
            uq = obj.trainDataSet.uq(1:ndat);
			phidat = obj.trainDataSet.phidat(1:ndat);
            
            umax = min(max(udat),1);
                
			% plot pred u
			[ax1,ax2] = obj.scatter(xdat, upredxdat.*phidat, '\phi u_{pred}');
            clim(ax2, [0, umax]);
            
			fname = 'fig_upredxdat.jpg';
			obj.savefig(fname);
			

			% plot data u
			[ax1,ax2] = obj.scatter(xdat, udat.*phidat, '\phi u_{dat}');
            clim(ax2, [0, umax]);
			fname = 'fig_udat.jpg';
			obj.savefig(fname);

			% plot error u
			err = (upredxdat - udat ).*phidat;
			[ax1,ax2] = obj.scatter(xdat, err, '\phi|u_{pred}-u_{dat}|');

			fname = 'fig_uprederr.jpg';
			obj.savefig(fname);


            % plot error noise
            warning('assuming xr and xdat same');
			noise = (udat - uq ).*phidat;
			[ax1,ax2] = obj.scatter(xdat, err, '\phi noise');

			fname = 'fig_noise.jpg';
			obj.savefig(fname);
		end

		function [ax1,ax2] = scatterres(obj)
            % scatter plot of residual 
            % note: not a good way of visualization, different t
			absres = abs(obj.upred{1}.resxdat);
        
			nres = obj.info.n_res_pts;
			phi = obj.trainDataSet.phiq(1:nres);
            

            xr = obj.upred{1}.xr(1:nres,:);

			[ax1,ax2] = obj.scatter(xr, absres.*phi, '\phi|res|');
			fname = 'fig_res.jpg';
			obj.savefig(fname);
        end


        function forward(obj, varargin)
            % forward FDM solve using inferred parameter
			rD = obj.upred{1}.rD;
			rRHO = obj.upred{1}.rRHO;
            
            % get training param
			[xdim,x0,zslice,dw,rho,tend,L,T,DW,RHO] = obj.trainDataSet.getvar('xdim','x0','zslice','dw','rho','tend','L','T','DW','RHO');
            [dwc,rhoc] = obj.trainDataSet.getvar('dwc','rhoc');
            
			dw2 = rD / (tend/T) * dwc;
			rho2 = rRHO /(tend/T) * rhoc;

            
            p.tend = tend;
            p.polar = false;
            p = parseargs(p, varargin{:});
            fprintf('run forward with tend = %g\n', p.tend);

            % forward using infered paramter
			obj.fwdmodel = GliomaSolver(xdim,  dw2, rho2 ,x0, p.tend, zslice);
            % forward using exact paramter
			obj.fwdmodele = GliomaSolver(xdim,  dw, rho ,x0, p.tend, zslice);

            if p.polar
                obj.fwdmodel.solvepolar()
                obj.fwdmodele.solvepolar()
            else
			    obj.fwdmodel.readmri(obj.atlas);
			    obj.fwdmodel.solve(varargin{:});

                obj.fwdmodele.readmri(obj.atlas);
			    obj.fwdmodele.solve('savesol',false);
            end
            obj.atlas = obj.fwdmodele.atlas; % copy atlas
    
            
			
		end

		function fwderr(obj)
            % error of fwd solution using infered param
			[ax1,ax2] = obj.atlas.imagesc2('df', obj.fwdmodel.uend.*obj.fwdmodel.phi);
			title(ax1,'\phi u_{fdm,pred}');
			obj.savefig('fig_fdm_upred.jpg');
			

			err =  abs(obj.fwdmodel.uend - obj.fwdmodele.uend).*obj.fwdmodel.phi;
			[ax1,ax2] = obj.atlas.imagesc2('df', err);
			title(ax1,'\phi |u_{fdm,pred} - u_{end}|');
			obj.savefig('fig_fdm_err.jpg');
			

        end

        function [ax1, ax2] = contourfdm(obj, level)
            [ax1, ax2] = obj.atlas.contoursc(obj.upred{1}.xdat, obj.upred{1}.upredxdat, level );
            
            ax3 = axes;
            contour(obj.atlas.gy, obj.atlas.gx, obj.fwdmodele.uend, level, 'y','LineWidth',2);
            
            set(ax3,'YDir','reverse')
            set(ax3,'color','none','visible','on')
            
            hLink = linkprop([ax3,ax2, ax1],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end


        function contourfwd(obj, level)
            [ax1,h1] = obj.atlas.plotbkgd(obj.atlas.df);
            ax1.Position(3) = ax1.Position(3)-0.1;
            
            ax2 = axes;
            contour(obj.atlas.gy, obj.atlas.gx, obj.fwdmodel.uend,level,'b','LineWidth',2);

            ax3 = axes;
            contour(obj.atlas.gy, obj.atlas.gx, obj.fwdmodele.uend,level,'r','LineWidth',2);
            
            set([ax2,ax3],'YDir','reverse')
            set([ax2,ax3],'color','none','visible','on')
            
            hLink = linkprop([ax3, ax2, ax1],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
        end


        function  axs = contourt(obj, k, level, i)
            u = obj.upred{k}.upredts(:,i);
            [~,fname,~] =  fileparts(obj.upred{k}.path);
            parts = split(fname,'_');
            optimizer  = parts{end};

            % interpolate solution at time from PINN
            tq = obj.upred{k}.ts * obj.trainDataSet.tend;
            ugrid = obj.fwdmodele.interpgrid(tq,'linear');
            phigrid = obj.fwdmodele.phi;
            
            
            [axbg, ~] = obj.atlas.plotbkgd(obj.atlas.df);
%             [axs(1), axs(2)] = obj.atlas.imagescfg(ugrid(:,:,i).*phigrid);
            %             ax3 = axes;
%             imagesc(ax3, obj.atlas.gx(1),obj.atlas.gy(1),ugrid(:,:,i).*phigrid);
%             [~,c2] = contour(obj.atlas.gy, obj.atlas.gx, ugrid(:,:,i).* phigrid , level,'r','LineWidth',1);

            
            axugrid = axes;
            [~,hContour] = contour(axugrid,obj.atlas.gx, obj.atlas.gy, ugrid(:,:,i).*phigrid, level);
            hContour.LineColor = "#D95319"; % red
            set(axugrid,'YDir','reverse','color','none','visible','on');
            % hacks to make contourf transparent
%             https://undocumentedmatlab.com/articles/customizing-contour-plots
%             drawnow;  % this is important, to ensure that FacePrims is ready in the next line!
%             hFills = hContour.FacePrims;  % array of TriangleStrip objects
%             [hFills.ColorType] = deal('truecoloralpha');  % default = 'truecolor'
%             for idx = 1 : numel(hFills)
%                hFills(idx).ColorData(4) = 150;   % default=255
%             end

%             hContour.LineColor = "#D95319"; % red
%             cugrid.FaceAlpha = 0;

            % interpolate scattered data to grid
            nres = obj.info.n_res_pts;
			phiq = obj.trainDataSet.phiq(1:nres);
            X = obj.upred{k}.xr;
            F = scatteredInterpolant(X(:,2), X(:,3), u, 'linear','none');
            uq = F(obj.atlas.gx, obj.atlas.gy);
            uq(isnan(uq)) = 0;

            % plot contour line
%             [ax1, ax2, c1] = obj.atlas.contour('df', uq, level);
            hold on
            [cpinn,hpinn] = contour(axugrid,obj.atlas.gx, obj.atlas.gy, uq, level);
            clabel(cpinn,hpinn,'LabelSpacing',100000);
            hpinn.LineColor = "#EDB120"; % yellow
%             set(axuq,'YDir','reverse','color','none','visible','on');
            set(axugrid,'YDir','reverse','color','none','visible','on');
            hLink = linkprop([ axugrid axbg],{'XLim','YLim','Position','DataAspectRatio'});
            hLink.Targets(1).DataAspectRatio = [1 1 1];
            
            levelstr = sprintf('%g ',level);
            title(axbg, sprintf('%s, contour %s, t = %g',optimizer, levelstr, tq(i)));
			fname = sprintf('fig_contourt_%s_t%g.jpg',optimizer, tq(i));
			obj.savefig(fname);
            
            axs = [axbg, axugrid];

        end

        function contourts(obj, pat, level, ts)
            % pat = pattern for file
            paths = cellfun(@(x) x.path, obj.upred, 'UniformOutput', false);
            k = find(contains(paths, pat));
            for ti = ts
                obj.contourt(k, level, ti);
            end
        end


        function residualscatter(obj, i)
            tq = obj.upred{3}.ts * obj.trainDataSet.tend;
            maxres = max(obj.upred{3}.rests(:));
            minres = min(obj.upred{3}.rests(:));
            [ax1,ax2]=obj.atlas.scatter('df', obj.upred{3}.xr, 6, obj.upred{3}.rests(:,i));
%             clim([minres maxres]);
            disp([minres maxres]);
            title(ax1, sprintf('residual t = %g',tq(i)));
        end
   end
end