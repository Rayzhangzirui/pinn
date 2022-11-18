classdef MriDataSet<DataSet
    properties
        modeldir
        mods
        dats
        infos % tag for each mod
        savefig
        visdir
        box
        range % used for colormap
        id % patient id
    end
   
    methods
        function obj = MriDataSet(varargin)
            p = inputParser;
            addParameter(p,'modeldir','',@ischar);
            addParameter(p,'visdir','',@ischar); % visualization dir
            addParameter(p,'mods',{},@iscell);
            addParameter(p,'dats',{},@iscell);
            addParameter(p,'savefig',true,@islogical);
            
            parse(p,varargin{:});
            
            % handel path
            obj.modeldir = p.Results.modeldir;
            assert(exist(obj.modeldir, 'dir')==7,'dir does not exist');
            fs = dir(fullfile(obj.modeldir,'*.nii.gz')); % file paths
            [~,obj.id,~]=fileparts(obj.modeldir);
            
            % visualization
            obj.visdir = p.Results.visdir;
            obj.savefig = p.Results.savefig;

            % which mods are readed
            readmods = p.Results.mods;
            
            j = 1;
            for i = 1:length(fs)
                fname = split(fs(i).name,{'.'});
                parts = split(fname{1},{'_'});
                whichmod = parts{end};

                % if isempty, read all, 
                if ~isempty(readmods) && ~contains(whichmod,readmods)
                    continue;
                end
                
                obj.mods{j} = whichmod;

                path = fullfile(fs(i).folder,fs(i).name);
                dat = MRIread(path);
                obj.dats{j}  = dat.vol;
                obj.infos{j}  = whichmod;
                
                % originally, seg, 1 = necrosis, 2 = edema, 4 = tumor
                if strcmp(whichmod,'seg')
                    % now, 1 = edema, 2 = tumor, 3 = necrosis
                    dat.vol(dat.vol==1)=6;
                    obj.dats{j} = dat.vol;
                    obj.getBox();
                end

                j = j + 1; % inc counter
            end 
        end %end of constructor
        

        function i = getidx(obj,mod)
            i = find(ismember(obj.mods,mod));
        end
        

        function [dat,i] = get(obj,mod)
            i = obj.getidx(mod);
            dat = {};
            if isempty(i)
                fprintf('%s not found\n',mod);
                return;
            end
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
        
        function getuadc(obj,minadc,maxadc)
            % different method to compute u from adc
            seg = obj.get('seg');
            adc = obj.get('md');
            
            adc = threshold(adc, minadc,maxadc);
            adc(seg<1)= nan;
            obj.append('adcseg',adc,'adc in seg');
            
            % linear transform in all region
            % msk = seg>1;
            % u = zeros(size(adc));
            % u(msk) = (obj.maxadc - adc(msk))/(obj.maxadc-obj.minadc);
            % obj.append('u1inv',u,'u linear, all region');
            
            % % quadratic in all region
            % msk = seg>1;
            % b = 0.25;
            % u = zeros(size(adc));
            % f = obj.minadc*b./adc(msk);
            % u(msk) = (1 + sqrt(1-4*f))/2;
            % obj.append('u1quad',u,'u quad, all region');
            
            % % linear transform in necrosis and tumor region
            % msk = seg>3;
            % u = zeros(size(adc));
            % u(msk) = (obj.maxadc - adc(msk))/(obj.maxadc-obj.minadc);
            % obj.append('u3inv',u,'u linear, tumor and necrosis region');
            
            % % quadratic transform
            % msk = seg>3;
            % u = zeros(size(adc));
            % f = obj.minadc*b./adc(msk);
            % u(msk) = (1 + sqrt(1-4*f))/2;
            % obj.append('u3quad',u,'u quad, tumor and necrosis region');
            
            % separate transform
            b = 1/4;
            u = zeros(size(adc));
            mske = seg==2; % indicator edema region
            mskt = seg>3; % indicator tumor region
            u(mskt) = (1 + sqrt(1-4*(minadc*b./adc(mskt))))/2;    
            u(mske) = minadc./adc(mske);

            u = imgaussfilt3(u,2,'FilterDomain','spatial');
            u(seg<1) = nan;
            obj.append('uadc',u,'u lin quad adc');
            
        end
        
        function stat = getBkgdStat(obj,mod,f)
            seg = obj.get('seg');
            dat = obj.get(mod);

            idx = dat>1e-6 & seg<1;
            med = median(dat(idx));
        end


        function getupetpatient(obj)
            seg = obj.get('seg');
            pet = obj.get('pet');
            
            bgidx = pet>1e-6 & seg<1;
            med = median(pet(bgidx));

            pet = pet - med;
            pet(pet<0) = 0;
            
            msk = seg>1;

            pet(msk) = rescale(pet(msk));
            obj.append('petpscale',pet,'pet re scaled');

            tumor = seg>3;
            edema = seg==2;
            u = zeros(size(pet));
            u(tumor) = (1 + sqrt(1-pet(tumor)))/2;
            u(edema) = (1 - sqrt(1-pet(edema)))/2;
            u = imgaussfilt3(u,2,'FilterDomain','spatial');
            u(seg<1) = nan;
            obj.append('upetp',u,'u quad pet 2');

        end

        function getupet(obj, lb, ub)
            seg = obj.get('seg');
            pet = obj.get('pet');
            pet = threshold(pet,lb, ub);
            u = zeros(size(pet));

            msk = seg>1;
            pet(~msk) = nan;
            obj.append('petseg',pet,'pet in seg');

            pet(msk) = rescale(pet(msk));
            obj.append('petscale',pet,'pet in seg scaled');

            tumor = seg>3;
            edema = seg==2;
            u(tumor) = (1 + sqrt(1-pet(tumor)))/2;
            u(edema) = (1 - sqrt(1-pet(edema)))/2;
            u = imgaussfilt3(u,2,'FilterDomain','spatial');
            u(seg<1) = nan;
            obj.append('upet',u,'u quad pet');
        end


        
        function getBox(obj,segname)
            if nargin<2
                segname = 'seg';
            end
            % find bounding box of tumor segmentation
            msk = obj.get(segname)>1;
            idx = find(msk);
            [i,j,k] = ind2sub(size(obj.get(segname)),idx);
            f = @(x) [min(x),max(x)];
            bx = f(i);
            by = f(j);
            bz = f(k);
            obj.box = [bx;by;bz];
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
      
        function hax = visual(obj,varargin)
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
            
            p.mod = 't2';
            p.slice = ceil(mean(obj.box(3,:)));
            p = parseargs(p,args{:});
            
            mod = p.mod;
            slice = p.slice;

            [dat,i] = obj.get(mod);
            
            pcolor(dat(:,:,slice));
            daspect([1 1 1]);
            shading flat;
            
            % discrete colorbar for seg
            if strcmp(mod,'seg')
                colormap(hax,parula(4));
                caxis([0 6]);
            end
            
            
            if startsWith(mod,'u')
                caxis([0 1]);
            end
            
            
            colorbar

            % title
            v = axis;
            posX = v(1) + 0.01 * (v(2) - v(1));
            posY = v(3) + 0.99* (v(4) - v(3));
            text( posX, posY, obj.infos{i},'Color','r',...
            'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'top');

            % turn off axis numbering
            set( hax, 'Visible', 'off' ) ;
        end
      
        function [fig,ha]= visualmods(obj,mods,nrow,slice,prefix)
            % visualize all 
            % prefix = plot prefig
            close all;            
            n = length(mods);
            if n == 0
                % default
                mods = obj.mods;
                prefix = 'all';
                n = length(mods);
                nrow = 2;
            end
            ncol = ceil(n/nrow);
            
            fig = figure('Position', [0 0 ncol*300 nrow*300]);
            
            ha = tight_subplot(nrow,ncol,0.02,0.02,0.02);
            
            for i = 1:length(mods)
                j = obj.getidx(mods{i});
                if isempty(j)
                    fprintf('%s not exist, skip\n',mods{i});
                    continue
                end
                obj.visual(ha(i),'mod',obj.mods{j},'slice',slice);
            end
            fname = sprintf('%s_slice%d',prefix,slice);
            obj.saveplot(fname,'-jpg','-m3');
        end
             
        function saveplot(obj,fname,varargin)
            % Save output
            if ~obj.savefig
                return
            end
                
            if( exist(obj.visdir,'dir') == 0 )
                sprintf('Output folder does not exist, creating it in: \n %s', obj.visdir)
                mkdir(obj.visdir)
            end
            
            ffname = sprintf('fig_%s_%s',obj.id,fname); % add patient number
            fp  = fullfile(obj.visdir,ffname);
            fprintf('save to %s',fp)
            export_fig(gcf, fp, varargin{:})
        end

        
        
        function histo(obj,mod,varargin)
            % histogram of adc
            dat = obj.get(mod);
            if isempty(dat)
                fprintf('skip histo');
                return;
            end
            seg = obj.get('seg');
            
            histogram(dat(dat>1e-6 & seg<1),'DisplayName','background',varargin{:});
            hold on;
            histogram(dat(seg==2),'DisplayName','edema',varargin{:});
            histogram(dat(seg==4),'DisplayName','tumor',varargin{:});
            histogram(dat(seg==6),'DisplayName','necrosis',varargin{:});
            hold off;
            legend('Location','best');
            
            fname = sprintf('fig_%s_hist_seg',mod);
            obj.saveplot(fname,'-jpg','-m3');
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
            
            fname = 'corrplot';
            obj.saveplot(fname,'-jpg');
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
            export_fig(gcf,fullfile(obj.visdir,'corrplot2'),'-png','-m2')
        end

        function p = plotline(obj, mod)
            % 1d plot of data
            [dat,idx] = obj.get(mod);
            seg = obj.get('seg');
            
            m = ceil(mean(obj.box,2));
            
            lseg = seg(:,m(2),m(3));
            y = dat(:,m(2),m(3));
            x = 1:length(lseg);
            
            segs = {'edema','tumor','necrosis'};
            colors = {"#0072BD","#D95319","#EDB120"};
            seglvl = [ 2 4 6];
            hold on
            for i = 1:3
                msk = lseg==seglvl(i);
                scatter(x(msk),y(msk),'filled','MarkerFaceColor',colors{i});
            end
            p = plot(x(lseg>1),y(lseg>1),'k','DisplayName',obj.infos{idx});
            legend(p,'Location','best');
            hold off;
        end


    end
end
