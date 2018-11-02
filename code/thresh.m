function [finalimg,final,finalclose] = thresh(img,a,b,c,d,e,f,g,h,k,l,m,n,o,p,q,r,t)

% Denoising
img= imgaussfilt(img,3);
% Extracting RBG Values
R = double(img(:,:,1));
G = double(img(:,:,2));
B = double(img(:,:,3));
% Extracting Hue and Saturation
imghsv = rgb2hsv(img);
H = imghsv(:,:,1); 
S = imghsv(:,:,2);
% Extracting YCbCr Spaces
yCbCr = rgb2ycbcr(img);
y = double(yCbCr(:,:,1));
Cb = double(yCbCr(:,:,2));
Cr = double(yCbCr(:,:,3));

% Thresholding RBG Space for Uniform Daylight or Under Flash Light or
% Daylight Lateral Illumination: Rule A
skinRGB = (R>a & G>b & B>c & (max(max(R,G),B)-min(min(R,G),B))>d ...
    & (R-G)>=e & R>G & R>B) | (R>f & G>g & B>h & (abs(R-G))<=t ...
    & R>B & G>B);

% Bounding Planes for Cb-Cr Subspace: Rule B
skinyCrCb = ~((Cb>=k & Cr<=l) & (Cb>=m & Cr<=n));

% Cutoff Levels or HS: Rule C
skinHS = (H>=o & H<=p) & (S>=q & S<=r);

final = skinRGB & skinyCrCb & skinHS;
finalclose = logical(imfill(final,'holes'));
finalimg = im2double(img).*final;
finalimg = im2uint8(finalimg);

end