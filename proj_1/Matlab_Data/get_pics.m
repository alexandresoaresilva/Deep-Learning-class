clc, clear, close all
kmeansdata = load('kmeansdata.mat');
onion = imread('onion.png');
imshow(onion)
cameraman = imread('cameraman.tif');
figure
imshow(cameraman)
imwrite(cameraman,'cameraman.tif');
imwrite(onion,'onion.png');