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
    scatter(X(index,3),X(index,4),color{i});
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
    scatter(X(index,3),X(index,4),color{i});
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
all_images = load('all_images.mat');
all_images = all_images.all_images; %remove struct

N_poses = 5:5:20;
N_feats = [16, 8, 4, 2];
%% 5 features, 16 features
no_matched_points = 0;
accuracy = 0;
train_accuracies = cell.empty();

N_feats = [16, 8, 4, 2];
    % for k=N_train

[no_matched_points, accuracy] = get_matches(all_images, N_poses(1), N_feats(1));

[no_matched_points1, accuracy1] = get_matches(all_images, N_poses(1), N_feats(4));
    
%     train_accuracies{1,1} = no_matched_points;
%     train_accuracies{1,2} = accuracy;


% for a=1:length(N_poses)
%     for k=1:length(N_feats)
%         for i=1:10 % i is the train object 
%             for j=1:10 % j is the test object
%                 [matchedPoints1, matchedPoints2] = ...
%                     get_match_pts(all_images, i,j, N_poses(a), N_feats(k));
%                 no_matched_points(i,j,k) = length(matchedPoints1.Scale);
%             end
%         end
%         
%         max_j = max(no_matched_points(i,:));
%         index_max = find(max_j == no_matched_points(i,:));
% 
%         % if train object has more features matched with the right test
%         if index_max == i
%             accuracy(k,i) = 1;
%         else
%             accuracy(k,i) = 0;
%         end
%     end
%     train_accuracies{a,1} = no_matched_points;
%     train_accuracies{a,2} = accuracy;
% end

function [no_matched_points, accuracy] =...
    get_matches(all_images, N_pose, N_feat)

    for i=1:10 % i is the train object 
        for j=1:10 % j is the test object
            [matchedPoints1, ~] = ...
                get_match_pts(all_images, i,j, N_pose, N_feat);
            no_matched_points(i,j) = length(matchedPoints1.Scale);
        end
        % after all matches have been made
        % if train object has more features matched with the right test
        max_j = max(no_matched_points(i,:));
        index_max = find(max_j == no_matched_points(i,:));

        if index_max == i
            accuracy(i) = 1;
        else
            accuracy(i) = 0;
        end
    end
    
    train_accuracies{1,1} = no_matched_points;
    train_accuracies{1,2} = accuracy;
end

% feature vectors
function [matchedPoints1, matchedPoints2] = ...
    get_match_pts(all_images, obj1,obj2, N_poses, N_feats)
    [feat_M1, SURF_M_concat1] =...
        get_feat_M_and_multi_SURF_points(all_images, obj1,...
                                         1, N_poses, N_feats);
    [feat_M2, SURF_M_concat2] =...
        get_feat_M_and_multi_SURF_points(all_images, obj2,...
                                        (N_poses + 1), 24, N_feats);

    indexPairs = matchFeatures(feat_M1, feat_M2,'Unique',1);

    matchedPoints1 = SURF_M_concat1(indexPairs(:,1));
    matchedPoints2 = SURF_M_concat2(indexPairs(:,2));
end
 
% this takes care of smaller
function [feat_M, SURF_M_concat] =...
    get_feat_M_and_multi_SURF_points(all_images, obj,...
                                    startPose, N_poses, N_feats)
    %gets no of features available
    [m,~] = size(all_images{obj,startPose,2});
    features  = 0;
    SURF_obj = 0;
    
    if m < N_feats %if fewer feats than the required are available
        features = all_images{obj,startPose,2};
        SURF_obj = all_images{obj,startPose,3};
    else
        features = all_images{obj,startPose,2}(1:N_feats,:);
        SURF_obj = all_images{obj,startPose,3}(1:N_feats);
    end
    
    feat_M = features;
    SURF_M_concat = SURF_obj;
    
    for i= (startPose+1):N_poses %concatenate all poses required
        
        [m,~] = size(all_images{obj,i,2});
        if m < N_feats %if fewer feats than the required are available
            features = all_images{obj,i,2};
            SURF_obj = all_images{obj,i,3};
        else
            features = all_images{obj,i,2}(1:N_feats,:);
            SURF_obj = all_images{obj,i,3}(1:N_feats);
        end
        %concatenate them vertically
        SURF_M_concat = vertcat(SURF_M_concat, SURF_obj);
        feat_M = [feat_M; features];
    end
end