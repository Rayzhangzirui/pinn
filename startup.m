set(0,'defaultfigurecolor',[1 1 1]);
set(0,'defaultAxesFontName', 'times')
set(0,'DefaultLineLineWidth', 1)
set(0,'defaultTextFontName', 'times')
set(0,'defaultAxesFontSize', 16)
set(0,'defaultTextInterpreter','latex')

addpath('/Users/zzirui/Documents/MATLAB/natsortfiles');
addpath('/Users/zzirui/Documents/MATLAB/arrow');
addpath('/Users/zzirui/Documents/MATLAB/export_fig-master');
addpath('/Users/zzirui/Documents/MATLAB/tight_subplot/');
addpath(genpath('/Users/zzirui/Documents/MATLAB/util'));

projdir = '/Users/zziruiadmin/projects/glioma/pinn';

projpath = @(x) fullfile(projdir,x)