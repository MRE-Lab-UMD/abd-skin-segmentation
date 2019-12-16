% %% Script to Rename
% clc;clear all;close all
% 
%% Input Directory
% d1_ = '../Input/Skin_Datasets/Dataset1_HGR/';
% d1ori= imageDatastore(strcat(d1_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d1mask= imageDatastore(strcat(d1_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% d2_ = '../Input/Skin_Datasets/Dataset2_TDSD/';
% d2ori= imageDatastore(strcat(d2_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d2mask= imageDatastore(strcat(d2_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% d3_ = '../Input/Skin_Datasets/Dataset3_Schmugge/';
% d3ori= imageDatastore(strcat(d3_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d3mask= imageDatastore(strcat(d3_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');

% d4_ = '../Input/Skin_Datasets/Dataset4_Pratheepan/';
% d4ori= imageDatastore(strcat(d4_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d4mask= imageDatastore(strcat(d4_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');

% d5_ = '../Input/Skin_Datasets/Dataset5_VDM/';
% d5ori= imageDatastore(strcat(d5_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d5mask= imageDatastore(strcat(d5_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% d6_ = '../Input/Skin_Datasets/Dataset6_SFA/';
% d6ori= imageDatastore(strcat(d6_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d6mask= imageDatastore(strcat(d6_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% d7_ = '../Input/Skin_Datasets/Dataset7_FSD/';
% d7ori= imageDatastore(strcat(d7_,'original_images'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d7mask= imageDatastore(strcat(d7_,'skin_masks'),'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% d8_ = '../Input/Skin_Datasets/Dataset8_Abdomen/train/';
% d8oritrain= imageDatastore(strcat(d8_,'skin_train2019'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d8masktrain= imageDatastore(strcat(d8_,'annotations'),'IncludeSubfolders',true,'LabelSource','foldernames');
% 
% d9_ = '../Input/Skin_Datasets/Dataset8_Abdomen/val/';
% d8orival= imageDatastore(strcat(d9_,'skin_val2019'),'IncludeSubfolders',true,'LabelSource','foldernames');
% d8maskval= imageDatastore(strcat(d9_,'annotations'),'IncludeSubfolders',true,'LabelSource','foldernames');

%% Output
% out = '../Input/Skin_Datasets/Coco_format/';
%% Uncommet as needed

%% Test
test = '../Input/Skin_Datasets/Test/';
k=1;
for i=1:100
        if (i<=78)
        imgd4ori = imread(d4ori.Files{i});
        imgd4mask = imread(d4mask.Files{i});
        imshowpair(imgd4ori, imgd4mask,'Montage');
        imwrite(imgd4ori,sprintf('%s%06d.jpeg',strcat(test,'skin_test2019/'),k));
        imwrite(imgd4mask,strcat(strcat(test,'annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
        k=k+1;
        end
        if ((i+3899)<=3999)
        imgd7ori = imread(d7ori.Files{i+3899});
        imgd7mask = imread(d7mask.Files{i+3899});
        imshowpair(imgd7ori, imgd7mask,'Montage');
        imwrite(imgd7ori,sprintf('%s%06d.jpeg',strcat(test,'skin_test2019/'),k));
        imwrite(imgd7mask,strcat(strcat(test,'annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
        k=k+1;
        end
        imgd8ori = imread(d8orival.Files{i});
        imgd8mask = imread(d8maskval.Files{i});
        imshowpair(imgd8ori, imgd8mask,'Montage');
        imwrite(imgd8ori,sprintf('%s%06d.jpeg',strcat(test,'skin_test2019/'),i+k));
        imwrite(imgd8mask,strcat(strcat(test,'annotations/'),sprintf('%s%06d','',i+k),'_skin_',sprintf('%s%06d','',i+k),'.png'));
        k=k+1;
end


%% Train
% for i=1:500
%     
%     if ((i+750)<=899)
%         imgd1ori = imread(d1ori.Files{i+750});
%         imgd1mask = imread(d1mask.Files{i+750});
%         imshowpair(imgd1ori, imgd1mask,'Montage');
%         imwrite(imgd1ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),k));
%         imwrite(imgd1mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
%         k=k+1;
%     end
%     
%     if ((i+500)<=555)
%         imgd2ori = imread(d2ori.Files{i+500});
%         imgd2mask = imread(d2mask.Files{i+500});
%         imshowpair(imgd2ori, imgd2mask,'Montage');
%         imwrite(imgd2ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),k));
%         imwrite(imgd2mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
%         k=k+1;
%     end
%     
%     if ((i+700)<=846)
%         imgd3ori = imread(d3ori.Files{i+700});
%         imgd3mask = imread(d3mask.Files{i+700});
%         imgd3mask(imgd3mask <30) =0;
%         imgd3mask(imgd3mask >30) = 255;
%         imshowpair(imgd3ori, imgd3mask,'Montage');
%         imwrite(imgd3ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),k));
%         imwrite(imgd3mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
%         k=k+1;
%     end
%     if ((i+250)<=285)
%         imgd5ori = imread(d5ori.Files{i+250});
%         imgd5mask = imread(d5mask.Files{i+250});
%         imgd5mask(imgd5mask <30) =0;
%         imgd5mask(imgd5mask >30) = 255;
%         imshowpair(imgd5ori, imgd5mask,'Montage');
%         imwrite(imgd5ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),k));
%         imwrite(imgd5mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
%         k=k+1;
%     end
%     if ((i+1000)<=1118)
%         imgd6ori = imread(d6ori.Files{i+1000});
%         imgd6mask = imread(d6mask.Files{i+1000});
%         imgd6mask(imgd6mask <30) =0;
%         imgd6mask(imgd6mask >30) = 255;
%         imshowpair(imgd6ori, imgd6mask,'Montage');
%         imwrite(imgd6ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),k));
%         imwrite(imgd6mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
%         k=k+1;
%     end
%     if ((i+3400)<=3899)
%         imgd7ori = imread(d7ori.Files{i+3400});
%         imgd7mask = imread(d7mask.Files{i+3400});
%         imshowpair(imgd7ori, imgd7mask,'Montage');
%         imwrite(imgd7ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),k));
%         imwrite(imgd7mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',k),'_skin_',sprintf('%s%06d','',k),'.png'));
%         k=k+1;
%     end
%     
% end

%% Val
% k=0;
% for i=1:756
%         imgd8ori = imread(d8orival.Files{i});
%         imgd8mask = imread(d8maskval.Files{i});
%         imshowpair(imgd8ori, imgd8mask,'Montage');
%         imwrite(imgd8ori,sprintf('%s%06d.jpeg',strcat(out,'val/skin_val2019/'),i+k));
%         imwrite(imgd8mask,strcat(strcat(out,'val/annotations/'),sprintf('%s%06d','',i+k),'_skin_',sprintf('%s%06d','',i+k),'.png'));
%      
% end
