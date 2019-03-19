clc;clear all;close all;
%% Setting Directory
% input='../Input/final/Phantom/';
input= '../Input/final/Phantom/Augmented/';
% cropped= '../Input/final/ICRA/skin level crop/wounds/wound';
% digitDatasetPath = fullfile('..\Input\final\detection\bandage\');
imds = imageDatastore(input,'IncludeSubfolders',true,'LabelSource','foldernames');

%% Augmentation Starts Here
k=1; % Naming Count for Augmented Images
goback=0;
for i=1:1:numel(imds.Files)
i
% Reading the image
img= imread(imds.Files{i});
[rows,cols, d]= size(img);
% Check if image is RGB or Gray scale, and convert it to RGB. IF any other input break code.
if ~(d==1 || d==3)
   disp(fullfile(' Invalid Image input at count: ', num2str(i)));
   disp (fullfile('Image Location is :', imds.Files{i}));
   disp('Check if Image is not a Gif');
   disp('Moving to Next image in 5 Seconds');
   pause(5);
   continue  
elseif( d==1)
%    Converting Gray scale to RGB
   img= img(:,:,[1 1 1]);    
end

%% Cropping the Right Image 
% close all
% img= imcrop (img);
% if (size(img,1)==0)
%     continue
% end
%% Resizing Images to [227 227]
img= imresize(img, [227,227]);
% imwrite(img,sprintf('%s%01d.jpg', cropped,k));k=k+1;
imwrite(img,sprintf('%s%01d.jpg', augmented,k));k=k+1;
%% Random Rotation
r= randi(3,1,1);
irot= imrotate(img,90*r);
% irot= imcrop(irot, [size(irot,2)/2,size(irot,1)/2,0.9*(min(size(irot,1),size(irot,2))),0.9*(min(size(irot,1),size(irot,2)))]);
irot= imresize(irot, [227,227]);
imwrite(irot,sprintf('%s%01d.jpg', augmented,k));k=k+1;
%% Random Translation
x= randi(30,1,1);y= randi(30,1,1);
% x=30,y=30;
lambda= 0.75;
itrans = imtranslate(img,[x, y]);
itrans= imcrop(itrans, [(113.5-lambda*113.5),(113.5-lambda*113.5),lambda*227,lambda*227]);
itrans= imresize(itrans, [227,227]);
imwrite(itrans,sprintf('%s%01d.jpg', augmented,k));k=k+1;
%% Random Mirroring about X & Y
flipr= randi(2,1,1);
iflip = flip(img,flipr);
iflip= imresize(iflip, [227,227]);
imwrite(iflip,sprintf('%s%01d.jpg', augmented,k));k=k+1;
% imshow(iflipx);
% imshow(iflipy);

%% Random change in Hue on any of the previous images
% Selecting Which image to take
rnum= randi(4,1,1);
allimg={img,irot,itrans,iflip};
imgout=allimg{rnum};
% Conversion to hsv and changing saturation and value.
hsvimg1 = rgb2hsv(imgout);
hsvimg2 = rgb2hsv(imgout);
% hsvimg(:,:,1) = hsvimg(:,:,1) * 0.9;
hsvimg1(:,:,2) = hsvimg1(:,:,2) * rand*1.5;
hueimg1 = hsv2rgb(hsvimg1);
hsvimg2(:,:,3) = hsvimg2(:,:,3) * rand*2;
hueimg2 = hsv2rgb(hsvimg2);
%Saving Two images here, one with different value and the other with
%different saturation
hueimg1= imresize(hueimg1, [227,227]);
imwrite(hueimg1,sprintf('%s%01d.jpg', augmented,k));k=k+1;
hueimg2= imresize(hueimg2, [227,227]);
imwrite(hueimg2,sprintf('%s%01d.jpg', augmented,k));k=k+1;

% % %% Visualizing all the Images Saved for each Case
% % subplot(3,2,1)
% % imshow(img);
% % title('Base image');
% % subplot(3,2,2)
% % imshow(irot);
% % title('Roatated image');
% % subplot(3,2,3)
% % imshow(itrans);
% % title('Translated image');
% % subplot(3,2,4)
% % imshow(iflip);
% % title('Flipped image');
% % subplot(3,2,5)
% % imshow(hueimg1);
% % title('Change in Value image');
% % subplot(3,2,6)
% % imshow(hueimg2);
% % title('Change in Saturation image');

%%
% answer = questdlg('Do you want to go back an image');
% if strcmp(answer,'Yes')
%     goback=1;
% elseif strcmp(answer, '')
%     goback=0;
% end

% ri= randi(3,1,1);
% iscale=imcrop(img,[round(size(img,1)/2),round(size(img,2)/2),ri*size(img,2),ri*size(img,2)]);
% imshow(iscale);

end

%% Renaming the Images and deleting Previous Versions
%  for i=1:1:numel(imds.Files)
%    currentimage = imread(imds.Files{i});   
%    imwrite(currentimage,sprintf('%s0%01d.jpg', input,i));
%    delete(imds.Files{i});
%     
%      
%  end