clc;clear all;close all;
%% Setting Directory
% input='../Input/final/Phantom/';
input_image1= './Input/allabdomen/train/skin_train2019/';
input_mask1= './Input/allabdomen/train/annotations/';
input_image = imageDatastore(input_image1,'IncludeSubfolders',true,'LabelSource','foldernames');
input_mask = imageDatastore(input_mask1,'IncludeSubfolders',true,'LabelSource','foldernames');
skin_augmented ='./Input/allabdomen/train/skin_augmented/';
annotations_augmented = './Input/allabdomen/train/annotations_augmented/';
augmentation1 = './Input/allabdomen/train/augmentation1/';

%% Augmentation Starts Here
k=756; % Naming Count for Augmented Images
goback=0;
for i=1:1:size(input_image.Files,1)

% Reading the image
img= imread(input_image.Files{i});
mask = imread(input_mask.Files{i});
% imshowpair(img,mask,'Montage');

%% Random Rotation
r= randi(3,1,1);
img_rot= imrotate(img,90*r);
mask_rot = imrotate(mask,90*r);
% imshowpair(img_rot,mask_rot,'Montage');
% irot= imcrop(irot, [size(irot,2)/2,size(irot,1)/2,0.9*(min(size(irot,1),size(irot,2))),0.9*(min(size(irot,1),size(irot,2)))]);
% irot= imresize(irot, [227,227]);
% imwrite(irot,sprintf('%s%01d.jpg', augmented,k));k=k+1;

%% Random Mirroring about X & Y
flipr= randi(2,1,1);
img_flip = flip(img,flipr);
mask_flip = flip(mask,flipr);
% imshowpair(img_flip,mask_flip,'Montage');
% imwrite(iflip,sprintf('%s%01d.jpg', augmented,k));k=k+1;
% imshow(iflipx);
% imshow(iflipy);
 
 %% Random change in Hue on any of the previous images
% Selecting Which image to take
rnum= randi(3,1,1);
allimg={img,img_rot,img_flip};
allmask={mask,mask_rot,mask_flip};

imgsel1=allimg{rnum};
masksel1=allmask{rnum};

rnum= randi(3,1,1);
imgsel2=allimg{rnum};
masksel2=allmask{rnum};

rnum= randi(3,1,1);
imgsel3=allimg{rnum};
masksel3=allmask{rnum};

% imshowpair(img_sel,mask_sel,'Montage');

% Conversion to hsv and changing saturation and value.
imghue1 = rgb2hsv(imgsel1);
imghue2 = rgb2hsv(imgsel2);
imghue3 = rgb2hsv(imgsel3);
maskhue1 = masksel1;
maskhue2 = masksel2;
maskhue3 = masksel3;

imghue1(:,:,1) = imghue1(:,:,1) *((1.2-0.8)*rand + 0.8);
imghue2(:,:,2) = imghue2(:,:,2) * ((1.2-0.6)*rand + 0.6);
imghue3(:,:,3) = imghue3(:,:,3) * ((1.2-0.9)*rand + 0.9);


imgafter1 = hsv2rgb(imghue1);
imgafter2 = hsv2rgb(imghue2);
imgafter3 = hsv2rgb(imghue3);

%Saving Two images here, one with different value and the other with
%different saturation

% % % %% Visualizing all the Images Saved for each Case
subplot(3,2,1)
imshowpair(img,mask,'Montage');
title('Base image');
subplot(3,2,2)
imshowpair(img_rot,mask_rot,'Montage');
title('Roatated image');
subplot(3,2,3)
imshowpair(img_flip,mask_flip,'Montage');
title('Flipped image');
subplot(3,2,4)
imshowpair(imgafter1,maskhue1,'Montage');
title('Change in hue image');
subplot(3,2,5)
imshowpair(imgafter2,maskhue2,'Montage');
title('Change in Saturation image');
subplot(3,2,6)
imshowpair(imgafter3,maskhue3,'Montage');
title('Change in value image');

%% Writting Images
saveas(gcf,sprintf('%s%05d.jpeg',augmentation1,i));
% Writing Rot Images
imwrite(img_rot,sprintf('%s%05d.jpeg',skin_augmented,i+k));
imwrite(mask_rot,strcat(annotations_augmented,sprintf('%s%05d','',i+k),'_skin_',sprintf('%s%05d','',i+k),'.png'));
k=k+1;

% Writing Flip Images
imwrite(img_flip,sprintf('%s%05d.jpeg',skin_augmented,i+k));
imwrite(mask_flip,strcat(annotations_augmented,sprintf('%s%05d','',i+k),'_skin_',sprintf('%s%05d','',i+k),'.png'));
k=k+1;

% Writing Hue Image 1
imwrite(imgafter1,sprintf('%s%05d.jpeg',skin_augmented,i+k));
imwrite(maskhue1,strcat(annotations_augmented,sprintf('%s%05d','',i+k),'_skin_',sprintf('%s%05d','',i+k),'.png'));
k=k+1;

% Writing Hue Image 2
imwrite(imgafter2,sprintf('%s%05d.jpeg',skin_augmented,i+k));
imwrite(maskhue2,strcat(annotations_augmented,sprintf('%s%05d','',i+k),'_skin_',sprintf('%s%05d','',i+k),'.png'));
k=k+1;

% Writing Hue Image 3
imwrite(imgafter3,sprintf('%s%05d.jpeg',skin_augmented,i+k));
imwrite(maskhue3,strcat(annotations_augmented,sprintf('%s%05d','',i+k),'_skin_',sprintf('%s%05d','',i+k),'.png'));
k=k+1;

end

%% Renaming the Images and deleting Previous Versions
%  for i=1:1:numel(imds.Files)
%    currentimage = imread(imds.Files{i});   
%    imwrite(currentimage,sprintf('%s0%01d.jpg', input,i));
%    delete(imds.Files{i});
%     
%      
%  end