classdef DataSet < dynamicprops
    
    methods
        function obj = DataSet(varargin)
            % take arg: var, .. 
            if nargin == 1 && isfile(varargin{1})
                fprintf('load from file')
                obj.load(varargin{1})
            else 
                for i = 1:nargin
                    obj.add(inputname(i),varargin{i});
                end
            end
        end

        function add(obj,prop,dat)
            % add by name value pair, modify if already exist
            if ~isprop(obj, prop)
                obj.addprop(prop);
            end
            if isnumeric(dat)
                dat = double(dat);
            end
            obj.(prop) = dat;
        end

        function addvar(obj,varargin)
            % add by variable
            % inputname(1) return object name
            % must be called directly, not by other function
            for i = 2:length(varargin)+1
                obj.add(inputname(i),varargin{i-1});
            end
        end

        function print(obj)
            % print all prop
            props = properties(obj);
            for i = 1:length(props)
                pname = props{i};
                fprintf('%s\n',pname);
            end
        end

        function save(obj,fname)
            % save properties
            props = properties(obj);
            for i = 1:length(props)
                pname = props{i};
                tmp.(pname) = obj.(pname);
            end
            save(fname,'-struct','tmp',props{:});
        end


        function copyprop(obj, a, varargin)
            % copy property from other object/struct a
            % obj.copyprop( struct_a, 'prop1', 'prop2')
            % if prop name not provided, copy all
            
            if isempty(a)
                warning('copyprop empty prop');
                return;
            end

            if isempty(varargin)
                fnames = fieldnames(a);
            else
                fnames = varargin;
            end

            for i = 1:length(fnames)
                pname = fnames{i};
                val = a.(pname);
                obj.add(pname, val);
            end
        end

        function load(obj, matfile)
            % load from matfile
            dat = load(matfile);
            fields = fieldnames(dat);
            for i=1:numel(fields)
                obj.add(fields{i}, dat.(fields{i}));
            end
        end


        function varargout = getvar(obj, varargin)
            % get field, assign to variables
            for i=1:numel(varargin)
                varargout{i} = obj.(varargin{i});
            end
        end

        
    end
end