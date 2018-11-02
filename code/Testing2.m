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

for lambda = 10:15
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
test = TR(:,1:9);
y =treee.predictFcn(test);

 tp = 0; tn = 0; fp = 0; fn = 0;

    for j=1:length(y)
        if y(j) == 0 && TR(j,10) == 0
            tn=tn+1;
        else if y(j) == 1 && TR(j,10) == 1
                tp = tp +1;
            else if y(j) == 0 && TR(j,10) == 1
                    fn = fn +1;
                else if y(j) == 1 && TR(j,10) == 0
                        fp = fp +1;
                    end
                end
            end
        end
    end

acc = [((tn+tp)/(tn+tp+fn+fp))*100, (fp/(tn+tp+fn+fp))*100, ...
    (fn/(tn+tp+fn+fp))*100];
