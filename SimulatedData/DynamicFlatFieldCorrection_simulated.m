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

addpath('../Vincent_code_func/');


% data
proj_noisy = h5read('synth_data.h5', '/proj_noisy');
flats = h5read('synth_data.h5', '/flats');
darks = h5read('synth_data.h5', '/darks');

nrImage = zeros(size(proj_noisy,2));
[det_horiz, projections, det_vert] = size(proj_noisy);

disp('load dark and flat fields:')

%load dark fields
disp('Load dark fields ...')
nrDark = size(darks, 2);
dark=zeros([det_horiz det_vert nrDark]);
for ii = 1:nrDark
    dark(:,:,ii)=double(squeeze(darks(:,ii,:)));
end
meanDarkfield = mean(dark,3);

%load white fields
nrWhite = size(flats, 2);
whiteVec = zeros([det_horiz*det_vert nrWhite]);

disp('Load white fields ...')
for ii=1:nrWhite
    tmp=double(squeeze(flats(:,ii,:)));
    whiteVec(:,ii) = tmp(:)-meanDarkfield(:);
end
mn = mean(whiteVec,2);

[M,N] = size(whiteVec);
Data = whiteVec;
% substract mean flat field
%Data = whiteVec - repmat(mn,1,N);
clear whiteVec dark
%% calculate Eigen Flat fields
% Parallel Analysis
nrPArepetions = 5;
disp('Parallel Analysis:')
[V1, D1, nrEigenflatfields]=parallelAnalysis(Data,nrPArepetions);
disp([int2str(nrEigenflatfields) ' eigen flat fields selected.'])
%%
%calculation eigen flat fields
eig0 = reshape(mn, det_horiz, det_vert);
EigenFlatfields = zeros([det_horiz det_vert 1+nrEigenflatfields]);
EigenFlatfields(:,:,1) = eig0;
for ii=1:nrEigenflatfields
    EigenFlatfields(:,:,ii+1) = reshape(Data*V1(:,N-ii+1), det_horiz, det_vert);
end
%% Filter Eigen flat fields
addpath('.\BM3D') 

display('Filter eigen flat fields ...')
filteredEigenFlatfields=zeros(det_horiz, det_vert, 1+nrEigenflatfields);

for ii=2:1+nrEigenflatfields
    display(['filter eigen flat field ' int2str(ii-1)])
    tmp=(EigenFlatfields(:,:,ii)-min(min(EigenFlatfields(:,:,ii))))/(max(max(EigenFlatfields(:,:,ii)))-min(min(EigenFlatfields(:,:,ii))));
    [~,tmp2]=BM3D(1,single(tmp));
    filteredEigenFlatfields(:,:,ii)=(tmp2*(max(max(EigenFlatfields(:,:,ii))) - min(min(EigenFlatfields(:,:,ii))))) + min(min(EigenFlatfields(:,:,ii)));
end
%filteredEigenFlatfields = EigenFlatfields; 

%% estimate abundance of weights in projections
% options output images
scaleOutputImages=  [0 1];          %output images are scaled between these values
% algorithm parameters
downsample=         2;              % amount of downsampling during dynamic flat field estimation (integer between 1 and 20)
nrPArepetions=      10;             % number of parallel analysis repetions

norm_proj_CFF = single(zeros(det_horiz, det_vert, projections));

meanVector=zeros(1,length(nrImage));
for ii=1:length(nrImage)
    disp(['conventional FFC: ' int2str(ii) '/' int2str(length(nrImage)) '...'])
    %load projection
    projection=double(squeeze(proj_noisy(:,ii,:)));
    
    tmp=(squeeze(projection)-meanDarkfield)./(EigenFlatfields(:,:,1));
    meanVector(ii)=mean(tmp(:));
    
    tmp(tmp<0)=0;
    tmp=-log(tmp);
    norm_proj_CFF(:,:,ii) = single(tmp);
    %tmp(isinf(tmp))=10^5;
    %tmp=(tmp-scaleOutputImages(1))/(scaleOutputImages(2)-scaleOutputImages(1));
    %tmp=uint16((2^16-1)*tmp);
    %imwrite(tmp,[outDIRFFC + outPrefixFFC + num2str(ii,numType) + fileFormat]);
end

%%
norm_proj_DFF = single(zeros(det_horiz, det_vert, projections));

xArray=zeros(nrEigenflatfields,length(nrImage));
for ii=1:length(nrImage)
    disp(['estimation projection ' int2str(ii) '/' int2str(length(nrImage)) '...'])
    %load projection
    projection=double(squeeze(proj_noisy(:,ii,:)));
    
    %estimate weights for a single projection
    x=condTVmean(projection,EigenFlatfields(:,:,1),filteredEigenFlatfields(:,:,2:(1+nrEigenflatfields)),meanDarkfield,zeros(1,nrEigenflatfields),downsample);
    xArray(:,ii)=x;
    
    %dynamic flat field correction
    FFeff=zeros(size(meanDarkfield));
    for  j=1:nrEigenflatfields
        FFeff=FFeff+x(j)*filteredEigenFlatfields(:,:,j+1);
    end
    
    tmp=(squeeze(projection)-meanDarkfield)./(EigenFlatfields(:,:,1)+FFeff);
    tmp=tmp/mean(tmp(:))*meanVector(ii);
    tmp(tmp<0)=0;
    tmp=-log(tmp);
    norm_proj_DFF(:,:,ii) = single(tmp);
    figure; imshow(squeeze(norm_proj_DFF(:,100,:)), [0, 1]);
    %tmp(isinf(tmp))=10^5;
    %tmp=(tmp-scaleOutputImages(1))/(scaleOutputImages(2)-scaleOutputImages(1));
    %tmp=uint16((2^16-1)*tmp);
    %imwrite(tmp,[outDIRDFFC + outPrefixFFC + num2str(ii,numType) + fileFormat]);
end

%save([outDIRDFFC '\' 'parameters.mat'], 'xArray')