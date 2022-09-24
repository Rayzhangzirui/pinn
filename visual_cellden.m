% visualize cell density

DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC)) % need MRIread
%%
datdir = '/Users/Ray/project/glioma/mridata/sfb05/' 
dat = MriDataSet('modeldir',datdir);

[~,~,bz] = dat.box();
mid  = ceil(bz(1) + diff(bz)/2);
slices = mid+5*[-2:1:2];
%%
[radc,sadc,mxadc,minadc] = dat.restrictadc();

%%
% dat.histo('adc1')
% dat.corrplot2('adc1',{'t1','t1c','t2'})
% dat.corrplot({'adc1','t1','t1c','t2','cbv','fla'})
%% get bounding box

%% visualize slices
for slice = slices
% %     dat.visualmods({},2,slice,'');
    dat.visualmods({'md','seg','adc1','u1','adc3','u3'},3,slice,'cmd');
end

%%

adcdat = dat.get('u1');
segdat = dat.get('seg');
mskdat = dat.get('smask');
%%
close all
plot(adcdat(120,:,mid))
hold on
plot(segdat(120,:,mid))
plot(mskdat(120,:,mid))

