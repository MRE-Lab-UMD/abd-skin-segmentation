%% Script to Rename
% clc;clear all;close all

%% Input Directory
dir_img = './Input/test_pratheepan/family/skin_test2019/';
dir_mask = './Input/test_pratheepan/family/annotations/';
img_input1 = './Input/dataset-pratheepan/testingdata/FamilyPhoto/';
% randfolder = './all abdomen/newtest/';
mask_input1 = './Input/dataset-pratheepan/groundtruth/FamilyPhoto/';
img_input = imageDatastore(img_input1,'IncludeSubfolders',true,'LabelSource','foldernames');
mask_input  = imageDatastore(mask_input1,'IncludeSubfolders',true,'LabelSource','foldernames');

% input1 = './abdomen images/1-450'  ;
% input  = imageDatastore(input1,'IncludeSubfolders',true,'LabelSource','foldernames');
% mask1 = './PixelLabelData_1/';
% mask_input  = imageDatastore(mask1,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Loop
k= 32;
% l=1;
 for i=1:1:100
%    gtimg = imread(strcat(img_input1,int2str(i),'.jpeg'));  
     gtimg = imread(img_input.Files{i});  
%      gtimg = imread(gTruth.DataSource.Source{i});
     
%    imshow(gtimg);
%    mask = imread(strcat(mask_input1,int2str(i),'_skin_',int2str(i),'.png'));
%      mask = imread(strcat(mask_input1,sprintf('%s%04d','',i),'.png'));
    maskimg = imread(mask_input.Files{i});
%     maskimg = maskimg(:,:,1);
    maskimg(maskimg>30) =255;
    maskimg(maskimg<30) =0;
%     maskimg = 255*imread(strcat(mask_input1,'Label_',int2str(i),'.png'));  
   imshowpair(gtimg,maskimg,'Montage');
   imwrite(gtimg,sprintf('%s%05d.jpeg', dir_img,i+k));
%    imwrite(gtimg,sprintf('%s%05d.jpeg',randfolder,i+k));
%     imshow(maskimg)
   imwrite(maskimg,strcat(dir_mask,sprintf('%s%05d','',i+k),'_skin_',sprintf('%s%05d','',i+k),'.png'));
%     l=l+1;
   % %    imwrite(currentimage,sprintf('%s0%01d.jpg', input1,i));
%    delete(img_input.Files{i})
%    delete(mask_input.Files{i})
%    
end