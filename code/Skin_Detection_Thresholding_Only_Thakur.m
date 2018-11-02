%% Anirudh Topiwala
% Skin Detection
% Main Script
clear all; close all; clc;

%% Reading All Images
imds_grndtruth = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\ground truth');
imds_testing = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\testing data');

%% Skin Detection + Accuracies Computation
acc = [];   %Matrix Containing Accuracies, False Positives and False Negatives Columns

for i = 1:length(imds_grndtruth.Files)
    img = imread(imds_testing.Files{i});
    gt = imread(imds_grndtruth.Files{i});
    [~,TH,~]= huelayer1(img);
    TH = uint8(TH);
    
    TH = reshape(TH,size(TH,1)*size(TH,2),1);
    gt = reshape(gt,size(gt,1)*size(gt,2),1);
    A = find(gt>0); gt(A) = 1; 
    
    tp = 0; tn = 0; fp = 0; fn = 0;

    for j=1:length(TH)
        if TH(j) == 0 && gt(j) == 0
            tn=tn+1;
        else if TH(j) == 1 && gt(j) == 1
                tp = tp +1;
            else if TH(j) == 0 && gt(j) == 1
                    fn = fn +1;
                else if TH(j) == 1 && gt(j) == 0
                        fp = fp +1;
                    end
                end
            end
        end
    end

acc = [acc; ((tn+tp)/(tn+tp+fn+fp))*100, (fp/(tn+tp+fn+fp))*100, ...
    (fn/(tn+tp+fn+fp))*100];

end

%% Function to apply first red hue segmentation
function [finalimg,final,finalclose]= huelayer1(img)
% Denoising
img= imgaussfilt(img,3);
% Extracting RBG Values
R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));
% Extracting Hue and Saturation
imghsv = rgb2hsv(img);
H = imghsv(:,:,1); 
S = imghsv(:,:,2);
% Extracting YCbCr Spaces
yCbCr = rgb2ycbcr(img);
y = double(yCbCr(:,:,1));
Cb = double(yCbCr(:,:,2));
Cr = double(yCbCr(:,:,3));

% Thresholding RBG Space for Uniform Daylight or Under Flash Light or
% Daylight Lateral Illumination: Rule A
skinRGB = (R>141 & G>234 & B>145 & (max(max(R,G),B)-min(min(R,G),B))>71 ...
    & (R-G)>=226 & R>G & R>B) | (R>70 & G>209 & B>88 & (abs(R-G))<=124 ...
    & R>B & G>B);

% Bounding Planes for Cb-Cr Subspace: Rule B
skinyCrCb = ~((Cb>=142 & Cr<=99) & (Cb>=50 & Cr<=200));

% Cutoff Levels or HS: Rule C
skinHS = (H>=54 & H<=176) & (S>=0.718 & S<=0.317);

final = skinRGB & skinyCrCb ;
finalclose = logical(imfill(final,'holes'));
finalimg = im2double(img).*final;
finalimg = im2uint8(finalimg);
imshow(finalimg)
end