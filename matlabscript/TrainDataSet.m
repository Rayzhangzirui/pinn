classdef TrainDataSet<DataSet
    

    methods

        function radialtestdat(obj)
            rtest = x2r(obj.xtest(:,2:3));
            scatter(rtest, obj.utest, 6, obj.xtest(:,1))
            colorbar
        end

        function xr3dist(obj,n)
            % 3d distribution of xr
            if nargin<2
                n = size(obj.xr,1);
            end
            scatter3( obj.xr(1:n,2), obj.xr(1:n,3), obj.xr(1:n,1), 12, obj.utest(1:n),'filled');
        end

        function x2dist(obj, xf, field,n)
            % 2d distribution of obj.(xf), color by obj.(field)
            X = obj.(xf);
            dat = obj.(field);
            if nargin<4
                n = size(X,1);
            end
            scatter( X(1:n,2), X(1:n,3), 12, dat(1:n) ,'filled');
        end

        function getdf(obj)
            % 3d distribution of xr
            df = obj.Pwmq * obj.dw + obj.Pgmq * obj.dg;
            obj.addvar(df);
            
        end


    end
end