classdef TrainDataSet<DataSet
    

    methods

        function radial(obj, xf, field,n)
            % 3d distribution of xr
            X = obj.(xf);
            dat = obj.(field);
            if nargin<4
                n = size(X,1);
            end

            rtest = x2r(X(:,2:3));
            scatter(rtest(1:n), dat(1:n), 6, X(1:n,1));
            colorbar
        end

        function dist3d(obj, xf, field, n)
            % 3d distribution of xr
            X = obj.(xf);
            dat = obj.(field);
            if nargin<4
                n = size(X,1);
            end
            scatter3( X(1:n,2), X(1:n,3), X(1:n,1), 12, dat(1:n),'filled');
        end

        function dist2d(obj, xf, field,n)
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