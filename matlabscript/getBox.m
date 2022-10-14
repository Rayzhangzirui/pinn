function [box,mid] = getBox(f, th, pad)
    % find the bounding box
    if nargin == 2
        pad = 0;
    end
    msk = f > th;
    idx = find(msk);
    [i,j,k] = ind2sub(size(f),idx);
    f = @(x) [min(x)-pad,max(x)+pad];
    bx = f(i);
    by = f(j);
    bz = f(k);
    box = [bx;by;bz];
    mid = ceil(mean(box,2));
end