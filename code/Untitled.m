%% Anirudh Topiwala
%% Wound Detection
%% Main Script
clear all; close all; clc;
%% Setting Directory
input= 'F:\UMD Summer\Wound Detection\Input\abdomen images\';
%% Reading Image
imds = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\ground truth');

string = '1.jpg';

cd '\\client\c$\Users\npatel26\Desktop\anirud/downloads';
img= imread(string);
cd '\\client\c$\Users\npatel26\Desktop\anirud/ground truth';
gt= imread(string);

for i=1:size(gt,1)
    for j=1:size(gt,2)
        if gt(i,j)>0
            gt(i,j)=1;
        end
    end
end
% img = imadjust(img1,stretchlim(img1),[]);
% imgkmeans=imcomplement(histeq(img));
% imgtext = 1-im2double(rgb2gray(img));
% imshowpair(img1,imgkmeans,'Montage');
% img= imcrop(img);
% if(~(isempty(img)))

%% Hue Layer Thresholding
cd '\\client\c$\Users\npatel26\Desktop\anirud';
[imghue,BM,BMclose]= huelayer1(img);
BM = uint8(BM);
BM = reshape(BM,size(BM,1)*size(BM,2),1);
gt = reshape(gt,size(gt,1)*size(gt,2),1);

tp = 0;
tn = 0;
fp = 0;
fn = 0;
for i=1:length(BM)
    if BM(i) == 0 && gt(i) == 0
        tn=tn+1;
    else if BM(i) == 1 && gt(i) == 1
            tp = tp +1;
        else if BM(i) == 0 && gt(i) == 1
                fn = fn +1;
            else if BM(i) == 1 && gt(i) == 0
                    fp = fp +1;
                end
            end
        end
    end
end
        
tp+tn+fp+fn