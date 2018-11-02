%% Anirudh Topiwala
% Skin Detection
% Main Script
clear all; close all; clc;

%% Reading All Images
imds_grndtruth = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\ground truth');
imds_testing = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\testing data');
TrainSkin = [];
TrainNS = [];
TrainSK = [];
TrainNSS = [];

for lambda = 1:15
image = imread(imds_testing.Files{lambda});
gt = imread(imds_grndtruth.Files{lambda});
 
Indices = [];
NK = [];
 for i=1:size(gt,1)
     for j=1:size(gt,2)
         if gt(i,j) > 0
             Indices = [Indices; i j];
         else NK = [NK; i j];
         end
     end
 end

 for i=1:size(Indices,1)
TrainSkin = [TrainSkin; image(Indices(i,1),Indices(i,2),1),...
    image(Indices(i,1),Indices(i,2),2),...
    image(Indices(i,1),Indices(i,2),3)];
 end

RGB = im2double(TrainSkin);
HSV = rgb2hsv(RGB);
YCBCR = rgb2ycbcr(RGB);
TrainSK = [TrainSK; RGB, HSV, YCBCR, ones(size(RGB,1),1)];


 for i=1:size(NK,1)
TrainNS = [TrainNS; image(NK(i,1),NK(i,2),1),...
    image(NK(i,1),NK(i,2),2),...
    image(NK(i,1),NK(i,2),3)];
 end

RGB = im2double(TrainNS);
HSV = rgb2hsv(RGB);
YCBCR = rgb2ycbcr(RGB);
TrainNSS = [TrainNSS; RGB, HSV, YCBCR, zeros(size(RGB,1),1)];
end

TR = [TrainSK; TrainNSS];