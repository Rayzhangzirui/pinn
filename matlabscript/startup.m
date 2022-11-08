set(0,'defaultfigurecolor',[1 1 1]);
set(0,'DefaultLineLineWidth', 1)

set(0,'defaultAxesFontName', 'times')
set(0,'defaultTextFontName', 'times')
set(0,'defaultAxesFontSize', 16)



set(0,'defaultTextInterpreter','tex')
set(0, 'defaultAxesTickLabelInterpreter','latex');
set(0, 'defaultLegendInterpreter','latex');

addpath('/Users/Ray/Documents/MATLAB/natsortfiles');
addpath('/Users/Ray/Documents/MATLAB/arrow');
addpath('/Users/Ray/Documents/MATLAB/export_fig-master');
addpath('/Users/Ray/Documents/MATLAB/tight_subplot/');
addpath(genpath('/Users/Ray/Documents/MATLAB/util'));

projdir = '/Users/Ray/projects/glioma/pinn';

DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC),'-end') % need MRIread
DIR_MRI = '/Users/Ray/project/glioma/jana/Atlas/anatomy/';
