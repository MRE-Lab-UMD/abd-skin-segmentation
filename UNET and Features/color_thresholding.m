%% Function to apply first red hue segmentation
function [finalimg,final,finalclose]= huelayer1(img)
%% Converting it into HSV after Denoising
img1= imgaussfilt(img,3);
imghsv= rgb2hsv(img1); imglab= rgb2lab(img1);
h=imghsv(:,:,1); s=imghsv(:,:,2); v=imghsv(:,:,3);
r=double(img(:,:,1));g=double(img(:,:,2));b=double(img(:,:,3));
R= r./(r+g+b);G= g./(r+g+b);
% figure
% imshowpair(img,imghsv,'Montage');f
% figure
% imshow(imghsv);
   yCbCr = rgb2ycbcr(img);
   y = im2uint8(yCbCr(:,:,1));
   Cb = im2uint8(yCbCr(:,:,2));
   Cr = im2uint8(yCbCr(:,:,3));
%    figure
%    imshow(yCbCr)
%% Thresholding hue values for red 
skinhsv= h>0.8 | h<0.2 ;
% figure
% imshow(huenew);
%% Thresholding yCRCB
skinycrcb = Cr>135 & Cb>85 & y>80 & Cr<=((1.5862*Cb)+20);
% skinycrcb = Cr>110 & Cb>95 & y>120;
% skinycrcb = (Cr>150| Cr<200) & (Cb>100 | Cb<150);

skinrgb=   r>95 &g>40 & b>20 & r>g &r>b & abs(r-g)>15;
skinrgbnorm= skinrgb & (R<0.465 | R>0.36) &  (G < 0.363 | G> 0.28);
final=(skinrgb) |(skinhsv & skinycrcb);

% figure
% subplot(3,3,1)
% imshow(img);
% title('Image');
% subplot(3,3,2)
% imshow(skinhsv);
% title('hue');
% subplot(3,3,3)
% imshow(skinycrcb);
% title('ycrcb');
% subplot(3,3,4)
% imshow(skinrgb);
% title('rgb');
% subplot(3,3,5)
% imshow(skinrgb| skinhsv);
% title('RGB and huenew');
% subplot(3,3,6)
% imshow(skinrgb| skinycrcb);
% title('RGB and ycrcb');
% subplot(3,3,7)
% imshow(final);
% title('final');
% subplot(3,3,8)
% imshow(skinrgbnorm);
% title('skinrgbnorm');
% subplot(3,3,9)    
% imshow(skinrgb  skinhsv);
% title('ycrcb and huenew');
% skin= huenew| skinycrcb;
%% Doing Morphological 
% BM1= imfill(huenew,'holes');
% se = strel('disk',5);
finalclose= logical(imfill(final,'holes'));
finalimg= im2double(img).*final;
finalimg=im2uint8(finalimg);
% figure
% imshowpair(finalimg,final,'Montage');
% BM3= bwareafilt(BM2,2);
% % imshow(BM3);
% newimg= im2uint8(im2double(img).*repmat(BM3,[1 1 3]));
% figure
% imshowpair(img,newimg,'Montage');

end