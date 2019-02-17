% clear, clc, close all
imgs = [3,4,5,6, 9,10, 12,13,14, 19];
pose = [0:11 60:71];
all_images = cell(length(imgs), length(pose),3);

% feats = cell(length(imgs), length(pose))

for i=1:length(imgs)
    feats = cell(1, length(pose));
    imgs = cell(1, length(pose));
    validPoints_dummy = cell(1, length(pose));
    for j=1:length(pose)
        filename = ['obj', num2str(i),'__', num2str(j),'.png'];
        I = imread(filename);
        imgs{j} = I;
        s_feat = detectSURFFeatures(I);
        [features,validPoints] = extractFeatures(I, s_feat);
%         validPoints_dummy{j} = validPoints.selectStrongest(16);
        validPoints_dummy{j} = validPoints;
        feats{j} = features;
    end 
    all_images(i,:,1) = imgs;
    all_images(i,:,2) = feats;
    all_images(i,:,3) = validPoints_dummy;
end

save('all_images.mat','all_images')
% indexPairs = matchFeatures(features1,features2);


