classdef Model
   % class to handel postprocessing data
   properties
      modeldir
      tag
      udat
      history
      predxr
      predt
      log
      info
   end
   
   methods
      function obj = Model(modeldir,varargin)
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
          
          if ~isempty(p.Results.udat)
              % if exact is provided
              obj.udat = readtable(p.Results.udat);
          end
          
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
                  [obj.log, obj.info] = readlog(path);
                  
                  fprintf('read %s\n',path);
                  continue

              end
                  
              if contains(fs(i).name, 'predt')
                  obj.predt = readtable(path,'ReadVariableNames',true);
                  fprintf('read %s\n',path);

                  continue
              end
              
              if contains(fs(i).name, 'predxr')
                  obj.predxr = readtable(path,'ReadVariableNames',true);
                  fprintf('read %s\n',path);
                  if p.Results.isradial
                      obj.predxr.r = sqrt(obj.predxr.x.^2 + obj.predxr.y.^2);
                  end
                  
                  continue
              end
          end
      end %end of constructor


       function sc = PlotLoss(obj,varargin)
        sc(1) = plot(obj.log.it, log10(obj.log.total),'DisplayName', [obj.tag ' total'],varargin{:});
        hold on;
        if ~all(obj.log.data == 0)
          sc(2) = plot(obj.log.it, log10(obj.log.res),'DisplayName', [obj.tag ' res'],varargin{:});
          sc(3) = plot(obj.log.it, log10(obj.log.data),'DisplayName', [obj.tag ' data'],varargin{:} );
        end
       end      
   end
end