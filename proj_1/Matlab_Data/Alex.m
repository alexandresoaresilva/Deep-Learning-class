clc, clear, close all
filt =@(n)1/n^2*ones(n);

%% 2D Fourier Space
% ============= 

%% 2D Fourier Space 2 (low pass filter)



%% onions convolutions


% conI

%% SURF features

I = imread('cameraman.tif');
s_feat = detectSURFFeatures(I);
[features,validPoints] = extractFeatures(I,s_feat);
figure
strongest = validPoints.selectStrongest(10);
imshow(I); hold on;
plot(strongest);