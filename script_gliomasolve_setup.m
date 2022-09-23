% tumor growth using diffused domain method
% work both in 3d and 2d
% use ndgrid instead of meshgrid

%% MRI data
DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC),'-end') % need MRIread
DIR_MRI = '/Users/Ray/project/glioma/jana/Atlas/anatomy/'
gm = MRIread( [DIR_MRI 'GM.nii']);
wm = MRIread( [DIR_MRI 'WM.nii']);
csf = MRIread( [DIR_MRI 'CSF.nii']);
%%
DIM = 3;
zslice = 99; % slice for visualization
Pwm = wm.vol;
Pgm = gm.vol;
Pcsf = csf.vol;
%%
if DIM == 2
%%% for 2d case, slice 3d MRI, then set zslice = 1, the third dimension is
%%% just 1. Filter with periodic padding make all z-derivative 0. So same as 2D
    Pwm = Pwm(:,:,zslice);
    Pgm = Pgm(:,:,zslice);
    Pcsf = Pcsf(:,:,zslice);
    zslice = 1;
end


%% solver parameters
tfinal = 150; %day
Dw = 0.13; % mm^2/day
Dg = Dw/10;
rho= 0.025; %0.025/day
x0 = [164 116];

h = 1; % spacial resolution, mm (caption figure 1)
epsilon = 3; % width of diffused domain

D = Pwm*Dw + Pgm*Dg; % diffusion coefficients


sz = [1 1 1];
sz(1:numel(size(Pwm))) = size(Pwm); % even when Pwm is 2d, 3rd dim is 1

% integer grid, horizontal is x, vertical y, only used to get u0
[gx,gy,gz] = ndgrid(1:sz(1),1:sz(2),1:sz(3)); 

ix = [x0 zslice]; % pixel index of initial tumor