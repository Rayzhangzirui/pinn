classdef DataSet < dynamicprops
    
    methods
        function obj = DataSet(varargin)
            % take arg: var, .. 
            for i = 1:nargin
                obj.add(inputname(i),varargin{i});
            end
        end

        function add(obj,prop,dat)
            % add by name value pair
            obj.addprop(prop);
            obj.(prop) = dat;
        end

        function addvar(obj,varargin)
            % add by variable
            % inputname(1) return object name
            for i = 2:length(varargin)+1
                obj.add(inputname(i),varargin{i-1});
            end
        end

        function print(obj)
            % print all prop
            props = properties(obj);
            for i = 1:length(props)
                pname = props{i};
                fprintf('%s\n',pname)
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
            for i = 1:length(varargin)
                pname = varargin{i};
                val = a.(pname)
                obj.add(pname, val)
            end
        end
        
    end
end