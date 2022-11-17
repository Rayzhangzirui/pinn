function dat = contourdat(levels,X,Y,Z)
% get the dat for each level
% dat.x{i} = x coord of data from levels(i)

coord = contourc(X,Y,Z, levels);

zs = [];
ns = [];
x = {};
y = {};
k = 1;
j = 1;
while k <= size(coord,2)
    z = coord(1,k);
    zs(end+1) = z;
    n = coord(2,k);
    ns(end+1) = n;

    idx = [k+1:k+n];
    x{end+1} = coord(1,idx); 
    y{end+1} = coord(2,idx);
    k = k+1+n;
end

% sort according to height
[dat.zs, idx] = sort(zs);
dat.ns = ns(idx);
dat.zsunique = unique(dat.zs);

dat.x = x(idx);
dat.y = y(idx);

end