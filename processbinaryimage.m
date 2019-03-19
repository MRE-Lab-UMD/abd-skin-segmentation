%% Script to Rename
% clc;clear all;close all

%% Input Directory
dir_img = './all abdomen/images/';
dir_mask = './all abdomen/skin_maksed_renamed/';
img_input1 = './all abdomen/train/skin_train2019/';
% randfolder = './all abdomen/newtest/';
mask_input1 = './all abdomen/train/annotations/';
img_input = imageDatastore(img_input1,'IncludeSubfolders',true,'LabelSource','foldernames');
mask_input  = imageDatastore(mask_input1,'IncludeSubfolders',true,'LabelSource','foldernames');

% input1 = './abdomen images/1-450'  ;
% input  = imageDatastore(input1,'IncludeSubfolders',true,'LabelSource','foldernames');
% mask1 = './PixelLabelData_1/';
% mask_input  = imageDatastore(mask1,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Loop
k= 769;
l=1;
 for i=11:1:14
%    gtimg = imread(strcat(img_input1,int2str(i),'.jpeg'));  
%      gtimg = imread(img_input.Files{i});  
%      gtimg = imread(gTruth.DataSource.Source{i});
     
%    imshow(gtimg);
%    mask = imread(strcat(mask_input1,int2str(i),'_skin_',int2str(i),'.png'));
%      mask = imread(strcat(mask_input1,sprintf('%s%04d','',i),'.png'));
    maskimg = imread(mask_input.Files{i});
%     maskimg = maskimg(:,:,1);
%     maskimg(maskimg>10) =255;
%     maskimg(maskimg<10) =0;
%     maskimg = 255*imread(strcat(mask_input1,'Label_',int2str(i),'.png'));  
%    imshowpair(gtimg,maskimg,'Montage');
%    imwrite(gtimg,sprintf('%s%05d.jpeg', dir_img,i+k));
%    imwrite(gtimg,sprintf('%s%05d.jpeg',randfolder,i+k));
%     imshow(maskimg)
%    imwrite(maskimg,strcat(dir_mask,sprintf('%s%05d','',l+k),'_skin_',sprintf('%s%05d','',l+k),'.png'));
    l=l+1;
   % %    imwrite(currentimage,sprintf('%s0%01d.jpg', input1,i));
%    delete(img_input.Files{i})
%    delete(mask_input.Files{i})
%    
end