% visualize second deata sed

DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC)) % need MRIread
datdir = sdir('/Users/Ray/project/glioma/mridata/for_UCI/sfb0*')
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