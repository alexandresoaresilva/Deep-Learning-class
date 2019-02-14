clc, clear, close all
filt =@(n)1/n^2*ones(n);
%% onion grayscale

I_gray = rgb2gray(imread('onion.png'));
imwrite(I_gray,'onion_gray.png');

%% 2D Fourier Space
% ============= 
I = imread('cameraman.tif');
If = fftshift(fft2(I));
If_log = log(1 + abs(If));
If_disp = If_log/max(If_log(:));
subplot(121)
imshow(I,[]);
subplot(122)
imshow(If_disp,[]);
%% 2D Fourier Space 2 (low pass filter)
radius = [5 10 25 50];
[x,y] = meshgrid(-128:127,-128:127);
Z = sqrt((x.^2) + (y.^2));

figure
for i=1:length(radius)
    R = radius(i); % Radius
    low_p_filt = zeros(size(Z));
    low_p_filt(Z < R) = 1;
%     Zl = conv2(Zl,filt(21),'same');
    If_low = If.*low_p_filt;

    % plot
    I_rec_low = ifft2(ifftshift(If_low));
    subplot(2,2,i);
    imshow(I_rec_low,[]);
    title(['radius: ',num2str(R)])
    xlabel('low pass mask');
end 
%% 2D Fourier Space 3 (high pass filter)
figure
for i=1:length(radius)
    high_p_filt = zeros(size(Z));
    R = radius(i); % Radius
    high_p_filt(Z > R) = 1;
%     zh = conv2(zh,filt(21),'same');
    If_high = If.*high_p_filt;
    I_rec_high = ifft2(ifftshift(If_high));

    % plot
    subplot(2,2,i);
    imshow(I_rec_high,[]);
    title(['radius: ',num2str(R)])
    xlabel('high pass mask');
end

%% onions convolutions

I = imread('coins.png');
v = [1 2 1];
sobel_x = [-v' [0 0 0]' v'];
sobel_y = [-v; 0 0 0; v];
int_filt = filt(5);

dI_dx = conv2(I, sobel_x)
dI_dy = conv2(I, sobel_y)
intI = conv2(I, int_filt)

figure
subplot(131)
imshow(I,[])
subplot(132)
imshow(dI_dx + dI_dy,[])

subplot(133)
imshow(intI,[])
% conI
%% clustering

load('kmeansdata.mat'); %loads
figure
for i=1:2
    subplot(2,2,i)
    scatter(X(:,i),X(:,i+2))
    xlabel(['feature ',num2str(i)]);
    ylabel(['feature ',num2str(i+2)]);
    title(['features ',num2str(i),...
        ' vs ',num2str(i+2)])
end

for i=3:4
    subplot(2,2,i)
    scatter(X(:,i-1), X(:,i))
    xlabel(['feature ',num2str(i-1)]);
    ylabel(['feature ',num2str(i)]);
    title(['features ',num2str(i-1),...
        ' vs ',num2str(i)])
end

k = 3;
idx = kmeans(X,k);
GMModel = fitgmdist(X,k)
GMModel = cluster(GMModel,X);

figure
subplot(1,2,1);
xlabel(['feature ',num2str(3)]);
ylabel(['feature ',num2str(4)]);
title('k-means')
color = {'r','b','g','k'};   
for i=1:4
    hold on
    index = find(idx == i);
    scatter(X(index,3),X(index,4),color{i})
end
hold off

subplot(1,2,2);
xlabel(['feature ',num2str(3)]);
ylabel(['feature ',num2str(4)]);
title('cluster')
color = {'r','b','g','k'};   
for i=1:4
    hold on
    index = find(GMModel == i);
    scatter(X(index,3),X(index,4),color{i})
end
hold off
%% SURF features

I = imread('cameraman.tif');
s_feat = detectSURFFeatures(I);
[features,validPoints] = extractFeatures(I,s_feat);
figure
strongest = validPoints.selectStrongest(10);
imshow(I); hold on;
plot(strongest);

%% after extracting images from Columbia U's dataset:
% it's not possible to collect 167 features for a lot of the pictures
% all_images(:,:,1): cell matrix, rows: each i_th is an obj
%                                 columns: each is a pose of the i_th obj
%                                 
% all_images(:,:,2): cell matrix, rows: each i_th is an obj
%                                 columns: each is a SURF features 
%                                          matrix of the of the i_th obj in
%                                           channel 1 (:,:,1)
% all_images(:,:,3): cell matrix, rows: each i_th is an obj
%                                 columns: each is a validPoints SURF 
%                                          obj of the of the i_th obj in
%                                           channel 1 (:,:,1)
% example of SURF features vector
%               Scale: [8×1 single]
%     SignOfLaplacian: [8×1 int8]
%         Orientation: [8×1 single]
%            Location: [8×2 single]
%              Metric: [8×1 single]
%               Count: 8

all_images = load('all_images.mat');
all_images = all_images.all_images; %remove struct
<<<<<<< HEAD
N_train = 5:5:20;
N_feats = {16, 8, 4, 2};
% feature vectors

feats = cell.empty();
feats_num= 0;

for i=1:5
   a = size(all_images{1,i,2});
   feats_num(i) = a(1);
end

max_num_of_feats = max(feats_num);
idx_max_feats = find(feats_num == max_num_of_feats);
idx_max_feats = idx_max_feats(1);

new_M = add_rows_to_feat_M( all_images{1,1,2}, max_num_of_feats);

%     feats{i} = all_images{1,1:5,2};

% feats_num(i) = max(e
% 
% indexPairs = matchFeatures(feats, f2);
% indexPairs2 = matchFeatures(feats, f3);

% % indexPairs = matchFeatures(f1,f2) ;
% matchedPoints1 = vpts1(indexPairs(:,1)); 
% matchedPoints2 = vpts2(indexPairs2(:,2));
function feat_M = add_rows_to_feat_M(feat_M, n_of_rows)
    [m,n] = size(feat_M);
    
    if n_of_rows ~= n
        num_of_rows_to_be_added = n_of_rows - m;

        for i=1:num_of_rows_to_be_added
            feat_M(i+m,:) = zeros(1,n);
        end
=======

M_feats = [16, 8, 4, 2];
N_train = [5 10 15 20];
% feature vectors
f1 = all_images{1,1,2};
f2 = all_images{1,2,2};
f3 = all_images{2,1,2};
% valid points obj
vpts1 = all_images{1,1,3};
vpts2 = all_images{1,2,3};

% indexPairs = matchFeatures(f1, f2);
% indexPairs2 = matchFeatures(f1, f3);
% indexPairs = matchFeatures(f1,f2) ;
% matchedPoints1 = vpts1(indexPairs(:,1));
% matchedPoints2 = vpts2(indexPairs2(:,2));

function M = create_feat_matrix(all_images,obj,n)
    dummy = cell(n,1)
    
    for i=1:n
        dummy{i} = all_images{obj,i,2};
        
>>>>>>> 5841aeb9b740fa17b258fdcf7c8533e0bb430766
    end
end