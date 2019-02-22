%% Script to Rename
% clc;clear all;close all

%% Input Directory
dir_img = './Input/coco-user/train/skin_train2019/' ;
dir_mask = './Input/coco-user/train/annotations/';
img_input1 = './Lydia_Segmented/images/' ;
mask_input1 = './Lydia_Segmented/ground_truth/';
img_input = imageDatastore(img_input1,'IncludeSubfolders',true,'LabelSource','foldernames');
mask_input  = imageDatastore(mask_input1,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Loop
k= 228;
 for i=1:1:size(img_input.Files,1)
   gtimg = imread(img_input.Files{i});  
   mask = imread(mask_input.Files{i});
%    imshowpair(gtimg,mask,'Montage');
   imwrite(gtimg,sprintf('%s%01d.jpeg', dir_img,k+i));
   imwrite(mask,strcat(dir_mask,int2str(k+i),'_skin_',int2str(k+i),'.png'));
%    imwrite(currentimage,sprintf('%s0%01d.jpg', input1,i));
%    delete(input.Files{i})
end