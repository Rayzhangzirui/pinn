% generate a list of files to combine to pdf
% cat ../matlabscripts/list_of_figs.txt | convert @- combine.pdf
files = dir('/Users/Ray/project/glioma/mridata/*/visnew/*')
[~,i] = sort({files.name});
files = files(i);
fid = fopen('./list_of_figs.txt','w');
for i = 1:length(files)
    if ~startsWith(files(i).name,'fig')
        continue
    end
    fprintf(fid, '%s\n', fullfile(files(i).folder,files(i).name));
end
fclose(fid)