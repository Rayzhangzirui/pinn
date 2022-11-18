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

%% histogram of pet values
allpet = zeros(240,240,155,28);
allseg = zeros(240,240,155,28);
alladc = zeros(240,240,155,28);
for i = 1:length(datdir)
    i
    dat(i) = MriDataSet('modeldir',datdir{i},'visdir',visdir,'mods',{'seg','pet','md'},'savefig',true);
    allpet(:,:,:,i) = dat(i).get('pet');
    allseg(:,:,:,i) = dat(i).get('seg');
    alladc(:,:,:,i) = dat(i).get('md');
end


%%

alldat = threshold(allpet,0,1e4);
close all
figure;
hold on;
histogram(alldat(allseg==2),'DisplayName','edema');
histogram(alldat(allseg==4),'DisplayName','tumor');
histogram(alldat(allseg==6),'DisplayName','necrosis');
legend('Location','best');

%%

adcth = [0, 3e-3]; % threshold
petth = [0, 1e4];

for i = 1:length(datdir)
% for i = 1:2
    dat(i) = MriDataSet('modeldir',datdir{i},'visdir',visdir,'mods',{'seg','pet','md'},'savefig',true);
    mid = ceil(mean(dat(i).box(3,:)));
    
    dat(i).getupet(petth(1), petth(2));

    dat(i).getuadc(adcth(1),adcth(2));

    [fig,ha] = dat(i).visualmods({'seg','adcseg','petseg','','uadc','upet'},2,mid,'');
    ha(2).CLim = adcth; ha(3).CLim = petth;
end

%% look at line plot
close all
p(1) = dat(i).plotline('upet')
hold on
p(2) = dat(i).plotline('petscale')
legend(p)
ax = gca;
ax.LineStyleOrder = {'-',':'}

dat(i).saveplot('line')
% [x,y,lseg] = dat(i).plotline('petscale')
% up = (1+sqrt(1-y/4))/2;
% um = (1-sqrt(1-y/4))/2;
% hold on
% plot(x,y/4)
% plot(x,up)
% plot(x,um)
% plot(x,y2)