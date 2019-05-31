% --------------------------------------------------------
% Copyright (c) Zhiding Yu
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to:
% 1. Generate the .bin edge labels that can be read by CASENet for training
% 2. Create caffe filelists for the generated data
% --------------------------------------------------------
function demo_preproc()

clc; clear; close all;

%% Setup Directories
dataRoot = '../data_orig';
genDataRoot = '../data_proc';
suffixImage = '_leftImg8bit.png';
suffixColor = '_gtFine_color.png';
suffixLabelIds = '_gtFine_labelIds.png';
suffixInstIds = '_gtFine_instanceIds.png';
suffixTrainIds = '_gtFine_trainIds.png';
suffixPolygons = '_gtFine_polygons.json';
suffixEdge = '_gtFine_edge.bin';

%% Setup Parameters
numCls = 19;
radius = 2;

%% Setup Parallel Pool
numWorker = 4; % Number of matlab workers for parallel computing
matlabVer = version('-release');
if( str2double(matlabVer(1:4)) > 2013 || (str2double(matlabVer(1:4)) == 2013 && strcmp(matlabVer(5), 'b')) )
    delete(gcp('nocreate'));
    parpool('local', numWorker);
else
    if(matlabpool('size')>0) %#ok<*DPOOL>
        matlabpool close
    end
    matlabpool open 8
end

%% Generate Output Directory
if(exist(genDataRoot, 'file')==0)
    mkdir(genDataRoot);
end

%% Generate Preprocessed Dataset
setList = {'train', 'val', 'test'};
for idxSet = 1:length(setList)
    setName = setList{idxSet};
    fidList = fopen([genDataRoot '/' setName '.txt'], 'w');
    cityList = dir([dataRoot '/leftImg8bit/' setName]);
    for idxCity = 3:length(cityList)
        cityName = cityList(idxCity, 1).name;
        if(exist([genDataRoot '/leftImg8bit/' setName '/' cityName], 'file')==0)
            mkdir([genDataRoot '/leftImg8bit/' setName '/' cityName]);
        end
        if(exist([genDataRoot '/gtFine/' setName '/' cityName], 'file')==0)
            mkdir([genDataRoot '/gtFine/' setName '/' cityName]);
        end
        fileList = dir([dataRoot '/leftImg8bit/' setName '/' cityName '/*.png']);

        % Generate and write data
        display(['Set: ' setName ', City: ' cityName])
        parfor_progress(length(fileList));
        parfor idxFile = 1:length(fileList)
            fileName = fileList(idxFile).name(1:end-length(suffixImage));
            % Copy image
            copyfile([dataRoot '/leftImg8bit/' setName '/' cityName '/' fileName suffixImage], [genDataRoot '/leftImg8bit/' setName '/' cityName '/' fileName suffixImage]);
            % Copy gt files
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixColor], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixColor]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixInstIds]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
            copyfile([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixPolygons], [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixPolygons]);
            if(~strcmp(setName, 'test'))
                % Transform label id map to train id map and write
                labelIdMap = imread([dataRoot '/gtFine/' setName '/' cityName '/' fileName suffixLabelIds]);
                trainIdMap = labelid2trainid(labelIdMap);
                imwrite(trainIdMap, [genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixTrainIds], 'png');
                % Transform color map to edge map and write
                edgeMapBin = seg2edge(labelIdMap, radius, [2 3]', 'regular'); % Avoid generating edges on "rectification border" (labelId==2) and "out of roi" (labelId==3)
                [height, width, ~] = size(trainIdMap);
                edgeMapCat = zeros(height, width, 'uint32');
                for idxCls = 1:numCls
                    segMap = trainIdMap == idxCls-1;
                    if(sum(segMap(:))~=0)
                        idxEdge = seg2edge_fast(segMap, edgeMapBin, radius, [], 'regular'); % Fast seg2edge by only considering binary edge pixels
                        edgeMapCat(idxEdge) = edgeMapCat(idxEdge) + 2^(idxCls-1);
                    end
                end
                fidEdge = fopen([genDataRoot '/gtFine/' setName '/' cityName '/' fileName suffixEdge], 'w');
                fwrite(fidEdge, edgeMapCat', 'uint32'); % Important! Transpose input matrix to become row major
                fclose(fidEdge);
            end
            parfor_progress();
        end
        parfor_progress(0);

		% Write file lists
        for idxFile = 1:length(fileList)
        	fileName = fileList(idxFile, 1).name(1:end-length(suffixImage));
        	if(~strcmp(setName, 'test'))
        		fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage ' /gtFine/' setName '/' cityName '/' fileName suffixEdge '\n']);
        	else
        		fprintf(fidList, ['/leftImg8bit/' setName '/' cityName '/' fileName suffixImage '\n']);
        	end
        end
    end
    fclose(fidList);
end
