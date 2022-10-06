% visualize cell density

DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC)) % need MRIread
%%

for i = 1:5
    datdir = sprintf('/Users/Ray/project/glioma/mridata/sfb0%d/',i)
    dat = MriDataSet('modeldir',datdir,'visdir','visnew');

    
    [~,~,bz] = dat.box();
    mid  = ceil(bz(1) + diff(bz)/2);
    
    
    dat.restrictadc(5e-4,0.003);
    dat.getu()
    
    
    
%     dat.histo('adccap')
%     dat.corrplot2('adc',{'t1','t1c','t2'})
%     dat.corrplot({'adc','t1','t1c','t2','cbv','fla'})
%     
%     dat.visualmods({},2,mid,'');    
    
    dat.visualmods({'md','seg','adc','u1inv','u1quad','ulinqua'},2,mid,'u');
    
end