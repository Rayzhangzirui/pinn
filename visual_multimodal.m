DIR_JANA_DATA_PROC = '/Users/Ray/project/glioma/jana/GliomaSolver/tools/DataProcessing';
addpath(genpath(DIR_JANA_DATA_PROC)) % need MRIread
%%
datdir = '/Users/Ray/project/glioma/mridata/sfb02/*.nii.gz';
paths = dir(datdir);
%%
vis.InputFolder  = fileparts(datdir);
vis.OutputFolder = fullfile(vis.InputFolder,'Visualisation/');

vis.Modalities = {paths.name};


vis.Colormap   = cell(1,length(vis.Modalities));
vis.Colormap(:) = {'gray'}

vis.nCols      = 4;
vis.nRows      = 2;


vis.bPositive  = 0; % keep only positive values  

vis.NameCutBy  = 0;

slices = 110;


%% get data and plot

for j = 1:length(slices)

    vis.Slice = slices(j);
    vis.OutputName   = ['slide_',num2str(vis.Slice)];

    ha = tight_subplot(2,4,0,0,0);

    for i = 1:length(vis.Modalities)
        

        inputPath = fullfile(vis.InputFolder,vis.Modalities{i});
        if( strcmp(inputPath(end-6:end), '.nii.gz'))
            inputPath=correctEmptySpaceInPathName(inputPath);
        end

        data      = MRIread(inputPath);

        if(vis.bPositive)
            data.vol(data.vol(:)<0) = 0;
        end

        baseName  = strsplit(vis.Modalities{i},'.');
        modName   = baseName{1};


        % plot on given position
        axes(ha(i))
        pcolor(data.vol(:,:,vis.Slice));
        daspect([1 1 1])
        shading flat;
    %     colormap(gca,vis.Colormap{i})
        colorbar

        % title
        v = axis;
        posX = v(1) + 0.1* (v(2) - v(1));
        posY = v(3) + 0.95* (v(4) - v(3));
        text( posX, posY, modName, 'FontSize', 15, 'FontWeight','bold','Color', [0.99,0.99,0.99],'Interpreter','none' );

        % turn off axis numbering
        set( gca, 'Visible', 'off' ) ;

    end

    % Save output
    if( exist(vis.OutputFolder,'dir') == 0 )
        sprintf('Output folder does not exist, creating it in: \n %s', vis.OutputFolder)
        mkdir(vis.OutputFolder)
    end

%     print(gcf,fullfile(vis.OutputFolder,vis.OutputName),'-djpeg')

end




