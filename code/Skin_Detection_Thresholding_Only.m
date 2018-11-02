%% Anirudh Topiwala
% Skin Detection
% Main Script
clear all; close all; clc;

% Reading All Images
imds_grndtruth = imageDatastore('../Input/user/groundtruth/');
imds_testing = imageDatastore('../Input/user/testingdata/');
% 
% imds_grndtruth = imageDatastore('../Input/dataset/groundtruth/FacePhoto/');
% imds_testing = imageDatastore('../Input/dataset/testingdata/FacePhoto/');

%% Skin Detection + Accuracies Computation
acc = [];   %Matrix Containing Accuracies, False Positives and False Negatives Columns

for i = 1:length(imds_grndtruth.Files)
    img = imread(imds_testing.Files{i});
    gt = imread(imds_grndtruth.Files{i});
    [~,TH,~]= huelayer1(img);
    TH = uint8(TH);
    
    TH = reshape(TH,size(TH,1)*size(TH,2),1);
    gt = reshape(gt(:,:,1),size(gt,1)*size(gt,2),1);
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
% Accuracy = [ Accuracy sensitivity specificity]
acc = [acc; ((tn+tp)/(tn+tp+fn+fp))*100, (tp/(tp+fn))*100, ...
    (tn/(tn+fp))*100];
    
end
meanacc = mean(acc,1);
disp ("the accuracy is");
meanacc(1)
disp ("the sensitivity is");
meanacc(2)
disp ("the specificty is");
meanacc(3)

%% Function to apply first red hue segmentation
function [finalimg,final,finalclose]= huelayer1(img)
% Converting it into HSV after Denoising
img1= imgaussfilt(img,3);
imghsv= rgb2hsv(img1); imglab= rgb2lab(img1);
h=imghsv(:,:,1); s=imghsv(:,:,2); v=imghsv(:,:,3);
r=double(img(:,:,1));g=double(img(:,:,2));b=double(img(:,:,3));
R= r./(r+g+b);G= g./(r+g+b);
yCbCr = rgb2ycbcr(img);
y = im2uint8(yCbCr(:,:,1));
Cb = im2uint8(yCbCr(:,:,2));
Cr = im2uint8(yCbCr(:,:,3));

% Thresholding hue values for red 
skinhsv= h>0.8 | h<0.2 ;

% Thresholding yCRCB
skinycrcb = Cr>135 & Cb>85 & y>80 & Cr<=((1.5862*Cb)+20);
skinrgb=   r>95 &g>40 & b>20 & r>g &r>b & abs(r-g)>15;
skinrgbnorm= skinrgb & (R<0.465 | R>0.36) &  (G < 0.363 | G> 0.28);

final=(skinrgb) |(skinhsv & skinycrcb);
finalclose= logical(imfill(final,'holes'));
finalimg= im2double(img).*final;
finalimg=im2uint8(finalimg);
end