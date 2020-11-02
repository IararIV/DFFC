%Dynamic intensity normalization using eigen flat fields in X-ray imaging
%--------------------------------------------------------------------------
%
% Script: Computes the conventional and dynamic flat field corrected
% projections of a computed tomography dataset.
%
% Input:
% Dark fields, flat fields and projection images in .tif format.
%
% Output:
% Dynamic flat field corrected projections in map
% 'outDIRDFFC'. Conventional flat field corrected projtions in
% map 'outDIRFFC'.
%
%More information: V.Van Nieuwenhove, J. De Beenhouwer, F. De Carlo, L.
%Mancini, F. Marone, and J. Sijbers, "Dynamic intensity normalization using
%eigen flat fields in X-ray imaging", Optics Express, 2015
%
%--------------------------------------------------------------------------
%Vincent Van Nieuwenhove                                        13/10/2015
%vincent.vannieuwenhove@uantwerpen.be
%iMinds-vision lab
%University of Antwerp

%% parameters
% data
addpath('../../Vincent_code_func/');

% Directory with raw dark fields, flat fields and projections in .tif format
projDir=  '~/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/sample/';        
% Directory where the DYNAMIC flat field corrected projections are saved
outDIRDFFC= '~/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/dark-field/';   
% Directory where the CONVENTIONAL flat field corrected projections are saved
outDIRFFC=  '~/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/flat-field/';    

fileFormat = '.tif';
numType = '%03d';

disp('load projections:')
sizeN = 2048;
sizeCut = 200;
projNo = 200;

projdata = single(zeros(sizeCut,sizeN,projNo));
for ii=1:projNo
 projection = single(imread([projDir 'IMAT00006388_PSI_cylinder_Sample_' num2str(ii-1,numType) fileFormat]));
 projdata(:,:,ii) = projection(1:sizeCut,:);
end

nrImage = zeros(size(projdata,3));
[det_horiz, det_vert, projections] = size(projdata);

disp('load dark and flat fields:')

%load dark fields
disp('Load dark fields ...')
dark=zeros([det_horiz det_vert 10]);
for ii=1:10
dark_temp = single(imread([outDIRDFFC 'IMAT00006385_PSI_cylinder_dark_' num2str(ii-1,numType) fileFormat]));
dark(:,:,ii) = dark_temp(1:sizeCut,:);
end

meanDarkfield = mean(dark,3);

%load white fields
% whiteVec = zeros([det_horiz*det_vert 59]);
%numType = '%04d'; 
%disp('Load white fields ...')
%for ii=1:29
%    whiteVec(:,ii) = double(reshape((imread([outDIRFFC 'Flat_' num2str(ii-1,numType) fileFormat])),det_horiz*det_vert,1)) - meanDarkfield(:);    
%end

% load corrected flat-fields
flats = h5read('/home/algol/Documents/DEV/DATA_TEMP/PSI_Phantom_White_Beam_Tomo/flats_corrected.h5', '/flats');
flats_s = flats(1:sizeCut,:,:);
whiteVec = reshape(flats_s, det_horiz*det_vert,59);

mn = mean(whiteVec,2);
clear flats flats_s
%%
[M,N] = size(whiteVec);
Data = whiteVec;
% substract mean flat field
%Data = whiteVec - repmat(mn,1,N);
%clear whiteVec dark
%% calculate Eigen Flat fields
% Parallel Analysis
nrPArepetions = 10;
disp('Parallel Analysis:')
[V1, D1, nrEigenflatfields]=parallelAnalysis(Data,nrPArepetions);
disp([int2str(nrEigenflatfields) ' eigen flat fields are selected.'])
%%
%calculation eigen flat fields
eig0 = reshape(mn, det_horiz, det_vert);
EigenFlatfields = zeros([det_horiz det_vert 1+nrEigenflatfields]);
EigenFlatfields(:,:,1) = eig0;
for ii=1:nrEigenflatfields
    EigenFlatfields(:,:,ii+1) = reshape(Data*V1(:,N-ii+1), det_horiz, det_vert);
end
% Filter Eigen flat fields
% addpath('.\BM3D') 

disp('Filter eigen flat fields ...')
filteredEigenFlatfields=zeros(det_horiz, det_vert, 1+nrEigenflatfields);

for ii=2:1+nrEigenflatfields
    display(['filter eigen flat field ' int2str(ii-1)])
    tmp=(EigenFlatfields(:,:,ii)-min(min(EigenFlatfields(:,:,ii))))/(max(max(EigenFlatfields(:,:,ii)))-min(min(EigenFlatfields(:,:,ii))));
    [~,tmp2]=BM3D(1, single(tmp), 0.25);
    filteredEigenFlatfields(:,:,ii)=(tmp2*(max(max(EigenFlatfields(:,:,ii))) - min(min(EigenFlatfields(:,:,ii))))) + min(min(EigenFlatfields(:,:,ii)));
end
%filteredEigenFlatfields = EigenFlatfields; 
%%
%% estimate abundance of weights in projections
% options output images
scaleOutputImages=  [0 1];          %output images are scaled between these values

norm_proj_CFF = single(zeros(det_horiz, det_vert, projections));

meanVector=zeros(1,length(nrImage));
for ii=1:length(nrImage)
    disp(['conventional FFC: ' int2str(ii) '/' int2str(length(nrImage)) '...'])
    %load projection
    projection=double(projdata(:,:,ii));
    
    tmp=(projection-meanDarkfield)./(EigenFlatfields(:,:,1));
    tmp(isnan(tmp))=0.0;
    meanVector(ii)= mean(tmp(:));
    
    tmp(tmp<0)=0;
    tmp=-log(tmp);
    norm_proj_CFF(:,:,ii) = single(tmp);
    tmp(isinf(tmp))=10^5;
    %tmp=(tmp-scaleOutputImages(1))/(scaleOutputImages(2)-scaleOutputImages(1));
    %tmp=uint16((2^16-1)*tmp);
    %imwrite(tmp,[outDIRFFC + outPrefixFFC + num2str(ii,numType) + fileFormat]);
end
figure; imshow(norm_proj_CFF(:,:,5), [-0.1 0.4]);
%%
norm_proj_DFF = single(zeros(det_horiz, det_vert, projections));
downsample=         4;              % amount of downsampling during dynamic flat field estimation (integer between 1 and 20)

xArray=zeros(nrEigenflatfields,length(nrImage));
for ii=1:length(nrImage)
    disp(['estimation projection ' int2str(ii) '/' int2str(length(nrImage)) '...'])
    %load projection
    projection=double(projdata(:,:,ii));
    
    %estimate weights for a single projection
    x=condTVmean(projection,EigenFlatfields(:,:,1),filteredEigenFlatfields(:,:,2:(1+nrEigenflatfields)),meanDarkfield,zeros(1,nrEigenflatfields),downsample);
    xArray(:,ii)=x;
    
    %dynamic flat field correction
    FFeff=zeros(size(meanDarkfield));
    for  j=1:nrEigenflatfields
        FFeff=FFeff+x(j)*filteredEigenFlatfields(:,:,j+1);
    end
    
    tmp=(projection-meanDarkfield)./(EigenFlatfields(:,:,1)+FFeff);
    tmp(isnan(tmp))=0.0;
    tmp=tmp/mean(tmp(:))*meanVector(ii);
    tmp(tmp<0)=0;
    tmp=-log(tmp);
    norm_proj_DFF(:,:,ii) = single(tmp);    
    tmp(isinf(tmp))=10^5;
    %tmp=(tmp-scaleOutputImages(1))/(scaleOutputImages(2)-scaleOutputImages(1));
    %tmp=uint16((2^16-1)*tmp);
    %imwrite(tmp,[outDIRDFFC + outPrefixFFC + num2str(ii,numType) + fileFormat]);
end
figure; imshow(squeeze(norm_proj_DFF(:,:,5)), [-0.1 0.4]);

%save([outDIRDFFC '\' 'parameters.mat'], 'xArray')