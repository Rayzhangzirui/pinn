function F = hmnLap(D, phi)
% harmonic laplacian

dim = length(size(D));
sz = size(D);

F = zeros(size(D));

parfor j = 1:prod(sz)
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
    for k = 1:dim
        for s = [1 -1]
            in = i;
            in(k) = in(k)+ s;
            
            if dim == 2
                jn = sub2ind(sz,in(1),in(2));
            else
                jn = sub2ind(sz,in(1),in(2),in(3));
            end
            
            hdf = 2/(1/D(j)+1/D(jn));
            dfs(end+1) = hdf;
            phis(end+1) = phi(jn);
        end
    end
    
    F(j) = sum(dfs.*phis)-sum(dfs)*phi(j);
    
end

end

