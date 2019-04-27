%% Script to Rename
clc;clear all;close all

%% Input Directory
input1= './bandage/';
input = imageDatastore(input1,'IncludeSubfolders',true,'LabelSource','foldernames');
%% Loop
 for i=1:1:size(input.Files,1)
   currentimage = imread(input.Files{i});  
%    currentimage= imresize(currentimage,[227 227]);
   imwrite(currentimage,sprintf('%s0%01d.jpg', input1,i));
   delete(input.Files{i})
end