classdef PostData<handle
   % class to handle postprocessing data
   properties
      modeldir
      tag
      udat  %
      upred % predictioin at residual points
      log % table of training log
      info  % information of model
   end
   
   methods
      function obj = PostData(modeldir,varargin)
          p = inputParser;
          addRequired(p,'modeldir',@ischar);
          addParameter(p,'udat','',@ischar);
          addParameter(p,'tag','',@ischar);
          addParameter(p,'dim',2,@isscalar);
          addParameter(p,'isradial',false,@islogical);
          parse(p,modeldir,varargin{:});
          
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
          end

          dat = load(fullfile(obj.modeldir,'upred_tfadam.mat'));
          obj.upred{1} = dat;
          dat = load(fullfile(obj.modeldir,'upred_scipylbfgs.mat'));
          obj.upred{2} = dat;

      end %end of constructor

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
           % remove scaling,
           scale = [T ones(1,xdim)*L];
           for i = 1:length(obj.upred)
               obj.upred{i}.xdat = obj.upred{i}.xdat .*scale + [0 x0];
               obj.upred{i}.xr = obj.upred{i}.xr .*scale + [0 x0];
           end
       end

       function x = getRelErr(obj,i,prop,exact)
           % get relative error
           relerr = @(x,xe) (x-xe)./xe;
           x = relerr(obj.upred{i}.(prop), exact);
       end

       function [fig, sc] = PlotInferErr(obj, rDe, rRHOe,varargin)
           % plot inference error
           relerr = @(x,xe) (x-xe)./xe;
           errD = relerr(obj.log.rD, rDe);
           errRHO = relerr(obj.log.rRHO, rRHOe);

           fig = figure;
           ts = sprintf('%s rel err\n adam rD = %0.2e, rRHO = %0.2e\n lbfg rD = %0.2e, rRHO = %0.2e',...
               obj.tag,...
           relerr(obj.upred{1}.rD, rDe),relerr(obj.upred{1}.rRHO, rRHOe),...
           relerr(obj.upred{2}.rD, rDe),relerr(obj.upred{2}.rRHO, rRHOe))
%            info = sprintf('afte'
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


   end
end