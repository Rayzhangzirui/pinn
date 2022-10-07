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
alldat = threshold(alladc,0,0.003);
close all
figure;
hold on;
histogram(alldat(allseg==2),'DisplayName','edema');
histogram(alldat(allseg==4),'DisplayName','tumor');
histogram(alldat(allseg==6),'DisplayName','necrosis');
legend('Location','best');

%%

for i = 5
    dat(i) = MriDataSet('modeldir',datdir{i},'visdir','vis','mods',{'seg','pet','md'},'savefig',false);
    dat(i).getupet(0, 1e4);
    mid = ceil(mean(dat(i).box(3,:)));


    dat(i).visualmods({'pet','seg','petseg','petscale','upet'},2,mid,'');
end

%%
close all
p(1) = dat(i).plotline('upet')
hold on
p(2) = dat(i).plotline('petscale')
legend(p)

% [x,y,lseg] = dat(i).plotline('petscale')
% up = (1+sqrt(1-y/4))/2;
% um = (1-sqrt(1-y/4))/2;
% hold on
% plot(x,y/4)
% plot(x,up)
% plot(x,um)
% plot(x,y2)


%%

for i = 6:9
    dat = MriDataSet('modeldir',datdir{i},'visdir','vis');

    
    [~,~,bz] = dat.box();
    mid  = ceil(bz(1) + diff(bz)/2);
    
    
    dat.restrictadc(5e-4,0.003);
%     dat.getu()
%     dat.visualmods({'cbv','fla','isodiff','md','pet','seg','t1','t1c','t2'},3,mid,'');
    dat.histo('pet');
    
    
    
%     dat.histo('adccap')
%     dat.corrplot2('adc',{'t1','t1c','t2'})
%     dat.corrplot({'adc','t1','t1c','t2','cbv','fla'})
%     
%     dat.visualmods({},2,mid,'');    
    
%     dat.visualmods({'md','seg','adc','u1inv','u1quad','ulinqua'},2,mid,'u');
    
end