classdef PostData<handle
   % class to handle postprocessing data
   properties
      modeldir
      tag
      udat  %
      upred % predictioin at residual points
      log % table of training log
      info  % information of model
      datinfo % info or training data
      history
   end
   
   methods
      function obj = PostData(modeldir,varargin)
          p = inputParser;
          addRequired(p,'modeldir',@ischar);
          addParameter(p,'udat','',@ischar);
          addParameter(p,'tag','',@ischar);
          addParameter(p,'dim',2,@isscalar);
          addParameter(p,'isradial',false,@islogical);
          addParameter(p,'dat_file','',@ischar);
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

              if startsWith(fs(i).name,'upred') && endsWith(fs(i).name,'mat')
                dat = load(path);
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
                  obj.datinfo = load(fp);
              else
                  fprintf('data file not found %s\n',fp);
              end
          end

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
           % remove scaling of (t,x)
           scale = [T ones(1,xdim)*L];
           for i = 1:length(obj.upred)
               if isfield(obj.upred{i},"xdat")
                   obj.upred{i}.xdat = obj.upred{i}.xdat .*scale + [0 x0];
               end
               obj.upred{i}.xr = obj.upred{i}.xr .*scale + [0 x0];
           end
       end

       function [fig, sc] = PlotInferErr(obj, rDe, rRHOe, varargin)
           % plot inference error
           
           errD = relerr(rDe, obj.log.rD);
           errRHO = relerr(rRHOe, obj.log.rRHO);

           fig = figure;
           ts = sprintf('%s rel err\n adam rD = %0.2e, rRHO = %0.2e\n',...
               obj.tag, relerr(obj.upred{1}.rD, rDe),relerr(obj.upred{1}.rRHO, rRHOe))
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