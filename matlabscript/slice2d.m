function f = slice2d(dat, vz, tk)
    % get 2d slice, dat might be (x,y,z),  (x,y,t) or (x,y,z,t)
    datdim = length(size(dat));
    
    if datdim==4
        if nargin==3
            f = dat(:,:,vz,tk);
        else
            f = dat(:,:,vz,end);
        end
        
    elseif datdim == 3 
        
        if vz>0
            % data is (x,y,z)
            f = dat(:,:,vz);
        else
            % data is (x,y,t)
            f = dat(:,:,tk);
        end
    else
        f = dat;
    end
end