function dat = contourdat(levels,x,y,Z)
% get the dat for each level
for i = 1:length(levels)
    coord = contourc(x,y,Z, [levels(i) levels(i)]);
    dat.x{i} = coord(1,2:end); % remove first element: level
    dat.y{i} = coord(2,2:end); % remove first element: num pts
end

end