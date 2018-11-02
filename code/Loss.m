function loss = Loss(X)

imds_grndtruth = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\ground truth');
imds_testing = imageDatastore('\\client\c$\Users\npatel26\Desktop\anirud\testing data');

%% Skin Detection + Accuracies Computation
loss = [];   %Matrix Containing Accuracies, False Positives and False Negatives Columns
a = X(1);
b = X(2);
c = X(3);
d = X(4);
e = X(5);
f = X(6);
g = X(7);
h = X(8);
k = X(9);
l = X(10);
m = X(11);
n = X(12);
o = X(13);
p = X(14);
q = X(15);
r = X(16);
t = X(17);

loss = 0;
for i = 1:1
    img = imread(imds_testing.Files{i});
    gt = imread(imds_grndtruth.Files{i});
    [~,TH,~]= thresh(img,a,b,c,d,e,f,g,h,k,l,m,n,o,p,q,r,t);
    TH = uint8(TH);
    
    TH = reshape(TH,size(TH,1)*size(TH,2),1);
    gt = reshape(gt,size(gt,1)*size(gt,2),1);
    A = find(gt>0); gt(A) = 1; 
    vec = 0.5*abs(TH - gt);
    loss = loss + sum(vec);
end

end