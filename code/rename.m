
%% Script to Rename
clc;clear all;close all

%% Input Directory
input= '../Input/dataset/groundtruth/FamilyPhoto/';

input = imageDatastore(input,'IncludeSubfolders',true,'LabelSource','foldernames');
output='../Input/dataset/groundtruth/FamilyPhoto/';
%% Loop
 for i=1:1:size(input.Files,1)
   currentimage = imread(input.Files{i});  
%    currentimage= imresize(currentimage,[227 227]);
   imwrite(currentimage,sprintf('%s%01d.jpg', output,i));
   delete(input.Files{i})
end