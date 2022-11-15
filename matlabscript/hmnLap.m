function F = hmnLap(D, u)
% apprx of div (D grad u)
% harmonic laplacian
% See Jana, GliomaSolve, ReactionDiffusionOperator.h



dim = length(size(D));
sz = size(D);

F = zeros(size(D));

parfor j = 1:prod(sz)
    % j is linear index, i is subscript
    if dim == 2
        [i1,i2] = ind2sub(sz,j);
        i = [i1,i2];
    else
        [i1,i2,i3] = ind2sub(sz,j);
        i = [i1,i2,i3];
    end

    % skip boundary
    if any(i==1)||any(i==sz)
        continue
    end
    
    dfs = [];
    phis = [];
    % go through each dimension, all nbr
    for k = 1:dim
        for s = [-1 1]
            in = i;
            in(k) = in(k)+ s;
            
            % in is subscritp of nbr, jn is linear index of nbr
            if dim == 2
                jn = sub2ind(sz,in(1),in(2));
            else
                jn = sub2ind(sz,in(1),in(2),in(3));
            end
            
            % harmonic mean of Dj and Djn
            hdf = 2/(1/D(j)+1/D(jn));
            dfs(end+1) = hdf;
            phis(end+1) = u(jn);
        end
    end
    
    F(j) = sum(dfs.*phis)-sum(dfs)*u(j);
    
end

end

