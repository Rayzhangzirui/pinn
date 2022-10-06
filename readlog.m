function [t,info] = readlog(path)
% read logging file
dats = [];
info = {};
if exist(path, 'file') ~= 2
    error('file not exist\n');
end


fid = fopen(path,'rt');
while ~feof(fid)
    tline = fgetl(fid);
    
    
    if startsWith(tline,'it') % get header
        tline = erase(tline,' ');
        header = split(tline,',');
        continue
    end

    s = split(tline,',');
    if all(isstrprop(s{1},'digit')) %get data
        dats(end+1,:) = cellfun(@str2num, s);
        continue
    else
        [tokens,matches] = regexp(tline,'(\S*) took (\d+\.?\d*)','tokens','match');
        if ~isempty(matches)
            info(end+1) = tokens;
        end
        
    end
    
end

fclose(fid);

t = array2table(dats,'VariableNames',header);


end

