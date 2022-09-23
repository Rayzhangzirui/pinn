classdef MriDataSet<handle
    properties
        modeldir
        mods
        dats
        infos
        savefig
        visdir
        WADC = 0.003
    end
   
    methods
        function obj = MriDataSet(varargin)
            p = inputParser;
            addParameter(p,'modeldir','',@ischar);
            addParameter(p,'mods',{},@iscell);
            addParameter(p,'dats',{},@iscell);
            parse(p,varargin{:});

            obj.modeldir = p.Results.modeldir;
            assert(exist(obj.modeldir, 'dir')==7,'dir does not exist');
            fs = dir(fullfile(obj.modeldir,'*.nii.gz')); % file paths


            for i = 1:length(fs)
                parts = split(fs(i).name,{'.','_'});
                obj.mods{i} = parts{3};
                path = fullfile(fs(i).folder,fs(i).name);
                dat = MRIread(path);
                obj.dats{i}  = dat.vol;
                obj.infos{i}  = obj.mods{i};
                
                % originally, seg, 1 = necrosis, 2 = edema, 4 = tumor
                if strcmp(obj.mods{i},'seg')
                    % now, 2 = edema, 4 = tumor, 6 = necrosis
                    dat.vol(dat.vol==1)=6;
                    obj.dats{i} = dat.vol;
                end
            end 
            
        obj.savefig = true;
        obj.visdir = fullfile(obj.modeldir,'visualization');

        end %end of constructor
        
        function i = getidx(obj,mod)
            i = find(ismember(obj.mods,mod));
        end
        
        function [dat,i] = get(obj,mod)
            i = obj.getidx(mod);
            assert(~isempty(i));
            dat = obj.dats{i};
        end
        
        
        function append(obj, mod, dat, info)
            % add data by mod
            i = obj.getidx(mod);
            if isempty(i)
                fprintf('append %s\n',mod);
                obj.mods{end+1}=mod;
                obj.dats{end+1} = dat;
                obj.infos{end+1} = info;
            else
                fprintf('re assign %s\n',mod);
                obj.mods{i}=mod;
                obj.dats{i} = dat;
                obj.infos{i} = info;
            end
        end
        
        function [cmd, u, maxmd, minmd] = restrictadc(obj)
            % restrict md to segmentation
            % valid = mask of tumor region
            %           valid = (obj.map('seg') == 1 | obj.map('seg') == 4);
            seg = obj.get('seg');
            obj.WADC = 0.003;
            
            for th = [1 3]
                % confine and threshold md
                msk  = seg>th;
                cmd = double(msk).* obj.get('md'); %confined md in msk region
                cmd(cmd>obj.WADC) = obj.WADC; % upper bound by free water adc
                cmd(~msk)=nan;
                % max and min adc restricted in seg
                maxmd = max(cmd(msk));
%                 minmd = min(cmd(msk));
                
                 minmd = 3e-4;

                info = sprintf('confined/capped md\n max %0.1e\n min %0.1e\n',maxmd,minmd);
                cmdname = ['adc' num2str(th)];
                obj.append(cmdname,cmd,info);
                

                % scaled
%                 f = minmd/cmd;
%                 u(cmd>obj.WADC) = 1;

                cmd(cmd<minmd) = minmd;
                b = 0.25;
                u = (1 + sqrt(1-4*(minmd*b/cmd).*double(msk)))/2;
%                 u = u/max(u(:));
                
%                 u*(1-u) is  b 1/adc
                
%                 utmp = minmd/cmd;
%                 u = utmp - min(utmp(:));
                
%                 u = 1 - minmd/cmd;
                
                u = u.*double(msk);
                
%                 u = (obj.WADC - cmd)/(obj.WADC-minmd).*double(msk);
                uname = ['u' num2str(th)];
                obj.append(uname,u,uname);
            end
        end
        
        function [bx,by,bz] = box(obj)
            % find bounding box of tumor segmentation
            msk = obj.get('seg')>1;
            idx = find(msk);
            [i,j,k] = ind2sub(size(obj.get('seg')),idx);
            % get bound with padding
            f = @(x) [min(x),max(x)];
            bx = f(i);
            by = f(j);
            bz = f(k);
        end
        
        % note: this seems to behave as handle, the original data is
        % chagned
%         function dataset = crop(obj, pad)
            % copy and crop
%             dataset = obj;
%             for i  = 1:length(obj.mods)
%                 mod = obj.mods{i};
%                 dat = obj.map(mod);
%                 dataset.map(mod) = dat(bx(1):bx(2),by(1):by(2),bz(1):bz(2));
%             end
%         end

      
      
        function visual(obj,varargin)
%             https://stackoverflow.com/questions/39365970/matlab-optional-handle-argument-first-for-plot-like-functions
            % Check the number of output arguments.
            nargoutchk(0,1);
            [ax, args, ~] = axescheck(varargin{:});

            % Get handle to either the requested or a new axis.
            if isempty(ax)
                hax = gca;
            else
                hax = ax;
            end
            axes(hax);
            
            p = inputParser;
            addOptional(p,'mod','t2',@(x) ischar(x));
            addOptional(p,'slice',100,@(x) isscalar(x));
            parse(p,args{:});
            
            mod = p.Results.mod;
            slice = p.Results.slice;

            [dat,i] = obj.get(mod);
            
            pcolor(dat(:,:,slice));
            daspect([1 1 1]);
            shading flat;
            
            % discrete colorbar for seg
            if strcmp(mod,'seg')
                colormap(hax,parula(4));
                caxis([0 6]);
            end
            
            if contains(mod,'cmd')
                caxis([0 obj.WADC]);
            end
            
            
            colorbar

            % title
            v = axis;
            posX = v(1) + 0.01 * (v(2) - v(1));
            posY = v(3) + 0.99* (v(4) - v(3));
            text( posX, posY, obj.infos{i},'Color', [1,1,1],...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'top');

            % turn off axis numbering
            set( hax, 'Visible', 'off' ) ;
        end
      
        
            
        function [fig,ha]= visualmods(obj,mods,nrow,slice,prefix)
            % visualize all 
            close all;            
            n = length(mods);
            if n == 0
                % default
                mods = {'cbv','fla','isodiff','md','seg','t1','t1c','t2'};
                prefix = 'all';
                n = 8;
                nrow = 2;
            end
            ncol = ceil(n/nrow);
            
            fig = figure('Position', [0 0 ncol*300 nrow*300]);
            
            ha = tight_subplot(nrow,ncol,0.02,0.02,0.02);
            
            for i = 1:length(mods)
                j = obj.getidx(mods{i});
                obj.visual(ha(i),obj.mods{j},slice);
            end
            obj.saveplot(sprintf('%s_slice%d',prefix,slice));
        end
             
        function saveplot(obj,fname)
            % Save output
            if ~obj.savefig
                return
            end
                
            if( exist(obj.visdir,'dir') == 0 )
                sprintf('Output folder does not exist, creating it in: \n %s', obj.visdir)
                mkdir(obj.visdir)
            end

            export_fig(gcf,fullfile(obj.visdir,fname),'-png','-m2')
        end
        
        function histo(obj,mod)
            % histogram of adc
            adc = obj.get(mod);
            seg = obj.get('seg');
            figure;
            hold on;
            histogram(adc(seg==2),'DisplayName','edema');
            histogram(adc(seg==4),'DisplayName','tumor');
            histogram(adc(seg==6),'DisplayName','necrosis');
            legend('Location','best');
            fname = 'fig_adc_hist_seg';
            export_fig(gcf,fullfile(obj.visdir,fname),'-png','-m2')
        end
        
        function corrplot(obj,mods)
            % correlation plot
            seg = obj.get('seg');
            msk = seg>1;
            n = length(mods);
            X = zeros(sum(msk(:)),n);
            for i = 1:n
                dat = obj.get(mods{i});
                X(:,i) = dat(msk);
            end
            t = array2table(X,'VariableNames',mods);
            corrplot(t);
            export_fig(gcf,fullfile(obj.visdir,'fig_corrplot'),'-png','-m2')
        end
        
        function corrplot2(obj,smod,tmods)
            % correlation plot of source mod and target mods in
            % segmentation
            seg = obj.get('seg');
            msk = seg>1;
            n = length(tmods);
            fig = figure('Position', [0 0 n*300 300]);
            ha = tight_subplot(1,n,0.1,0.2,0.1);
            
            sdat = obj.get(smod); % source data
            sdat = sdat(msk);
            for i = 1:n
                axes(ha(i));
                dat = obj.get(tmods{i});
                dat = dat(msk);
                scatter(sdat,dat,2,seg(msk),'filled')
                r = corrcoef(sdat,dat);
                xlabel(smod);
                ylabel(tmods{i});
                title(sprintf('r=%.2f',r(1,2)));
            end
            export_fig(gcf,fullfile(obj.visdir,'fig_corrplot2'),'-png','-m2')
        end

    end
end