%% Function to apply skin segmentation
% Input: Image in RGB
% Returns: Thresholded image

function [finalimg] = huelayer1(img)

% Converting it into HSV after Denoising
img1 = imgaussfilt(img,3);
imghsv = rgb2hsv(img1); imglab = rgb2lab(img1);
h = imghsv(:,:,1); s = imghsv(:,:,2); v = imghsv(:,:,3);
r = double(img(:,:,1)); g = double(img(:,:,2)); b = double(img(:,:,3));
R = r./(r+g+b); G = g./(r+g+b);
yCbCr = rgb2ycbcr(img);
y = im2uint8(yCbCr(:,:,1));
Cb = im2uint8(yCbCr(:,:,2));
Cr = im2uint8(yCbCr(:,:,3));

% Thresholding HSV (you have the flexibility to mofidy the thresholds)
skinhsv = h>0.8 | h<0.2 ;
% Thresholding YCbCr (you have the flexibility to mofidy the thresholds)
skinycrcb = Cr>135 & Cb>85 & y>80 & Cr<=((1.5862*Cb)+20);
% Thresholding RGB (you have the flexibility to mofidy the thresholds)
skinrgb = r>95 &g>40 & b>20 & r>g &r>b & abs(r-g)>15;

% Final threshold condition
final=(skinrgb) |(skinhsv & skinycrcb);

% Applying filter
finalimg = im2double(img).*final;
finalimg = im2uint8(finalimg);

end