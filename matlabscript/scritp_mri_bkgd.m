% visualize second deata set with pet
% collect all data set with pet
DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC)) % need MRIread
datdir = sdir('/Users/Ray/project/glioma/mridata/for_UCI/sfb*');
visdir = '/Users/Ray/project/glioma/mridata/for_UCI/vispet';

dirwithpet = {};
for i = 1:length(datdir)
    if contains(strjoin(sdir(datdir{i})),'pet')
        dirwithpet{end+1} = datdir{i};
    end
end

datdir = dirwithpet;

%%
adcth = [0, 3e-3]; % threshold
petth = [0, 1e4];

visdir = '/Users/Ray/project/glioma/mridata/for_UCI/vis_adc_histo';
for i = 1:length(datdir)
% for i = 1:2
    dat(i) = MriDataSet('modeldir',datdir{i},'visdir',visdir,'mods',{'seg','pet','md'},'savefig',true);
    mid = ceil(mean(dat(i).box(3,:)));
%     dat(i).getupet(petth(1), petth(2));
% 
%     dat(i).getupetpatient();
%     [fig,ha] = dat(i).visualmods({'seg','petseg','upet','upetp'},2,mid,'');
%     ha(2).CLim = petth;

    dat(i).histo('md','Normalization','pdf');
end
