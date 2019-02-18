clc, clear, close all
%% 3. Simple Image Classification Problem
% ========================================
% >>>>>> a. Feature detector and descriptor
I = imread('cameraman.tif');
s_feat = detectSURFFeatures(I);
[features,validPoints] = extractFeatures(I,s_feat);
figure
subplot(121);
strongest = validPoints.selectStrongest(10);
imshow(I); hold on;
plot(strongest);
title('cameraman.tif with 10 strongest features');
%% ================================================
% >>>>>> b.1. Creating the training feature matrix 
all_images = load('all_images.mat');
all_images = all_images.all_images; %removes struct
%% all_images: Columbia U's dataset, multidimensional cell array
%       all_images(:,:,1), 10 x 24: 10 rows = 10 objects
%                                   24 columns = 24 poses of the i_th obj
%       all_images(:,:,2), 10 x 24: 10 rows = represents 10 objects
%                                   24 columns = 24 SURF descriptors
%                                       (N features x 64 descriptors)                          
%       all_images(:,:,3), 10 x 24: 10 rows = represents 10 objects
%                                   24 columns = 24 SURF validPoints arrays
%                                       (N validPoints related to N 
%                                       features for each pose)                          

N_poses = 0:1:20;
N_feats = [2, 4, 8, 16];
%% ================================================
% >>>>>> b.2. Creating the training feature matrix 
% >>>>>> AND 
% >>>>>> AND c. Testing
% no_matched_points(i,j), rows: no. of TRAIN poses, 0 to 20, steps of 1
%                         columns: no. of selected features (from 2 to 16)
% accuracy(i,j), rows: no. of TRAIN poses, 0 to 20, steps of 1
%                columns: no. of selected features (from 2 to 16)
no_matched_points = cell.empty();
accuracy = 0;
 for i=1:length(N_poses)
     for j = 1:length(N_feats)
        [no_matched_points{i,j}, accuracy(i,j)] = get_matches(all_images,...
                                                    N_poses(i), N_feats(j));
     end
 end
 
 % plotting the accuracies
 accuracy = accuracy.*100;
%  figure
 subplot(122);
 plot(N_poses, accuracy(:,1));
 hold on
 grid on
 for i=2:4
    plot(N_poses, accuracy(:,i));
 end
 yticks(0:10:110);
 xticks(0:2:20);
 ylim([0 105]);
 xlim([0 20]);
 xlabel('Number of Training Images');
 ylabel('Accuracy(%)');
 legend({'2 features';'4 features';'8 features';'16 features'},...
     'Location','southeast');
 title({'Columbia University Image Library dataset: ';...
     'SURF feature matrices'' classification performance';...
     '10 different objects in 24 projections'});
 a = gcf;
 a.WindowState = 'maximized';
%% FUNCTIONS ===========================================================
% >>>> get_matches
%           inputs: 
%               all_images: 10 x 24 x 3 cell array 
%               N_poses: integer, no. of poses used for feature matrix
%               N_feats: integer, no. of features used in the feature matrix
%           outputs: 
%               no_of_matched_feats: number of matched features between 
%               two objects. used to calculate accuracy
%               accuracy: self explanary. Between 0 and 1.
function [no_matched_points, accuracy] = get_matches(all_images, N_poses, N_feats)
    % m == 10 different objets
    [m,~] = size(all_images(:,:,1));
    
    for i=1:m % i is the train object 
        for j=1:m % j is the test object
            no_matched_points(i,j) =...
                get_match_pts(all_images, i,j, N_poses, N_feats);
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
    accuracy = length(find(accuracy == 1))/length(accuracy);
end

% >>>> get_match_pts
%           inputs: 
%               all_images: 10 x 24 x 3 cell array
%               obj1: train object
%               obj2: test object
%               N_poses: integer, no. of poses used for feature matrix
%                        if 5 are used for feat. M, then 21 (24-5) will
%                        be used for the test matrix.
%               N_feats: integer, no. of features used in the feature matrix
%           outputs: 
%               no_of_matched_feats: number of matched features between 
%               two objects. used to calculate accuracy
function no_of_matched_feats =...
    get_match_pts(all_images, obj1,obj2, N_poses, N_feats)

  %train
    feat_M1 = get_feat_M(all_images, obj1,1, N_poses, N_feats);
  %test matrix
    feat_M2 = get_feat_M(all_images, obj2,(N_poses + 1), 24, N_feats);
    
    
    indexPairs = matchFeatures(feat_M1, feat_M2,'Unique',1);
    no_of_matched_feats = length(indexPairs);
end
 
% >>>> get_feat_M
%           inputs: 
%               all_images: 10 x 24 x 3 cell array
%               obj: object for which the feature matrix will be created
%               N_poses: integer, no. of poses used for feature matrix
%               N_feats: integer, no. of features used in the feature matrix
%           outputs: 
%               feat_M: feature matrix for obj with N_feats features and 
%                       composed with N_poses poses out of 24 total
function feat_M = get_feat_M(all_images, obj, startPose, N_poses, N_feats)
    %gets no of features available
    
    
    if ~N_poses % in the case of 0 features. feat matrix is N_feats x 64 
        % filled up with zeros
        feat_M = single(zeros(N_feats,64));
    else
        [m,~] = size(all_images{obj,startPose,2});
        features  = 0;
        if m < N_feats %if fewer feats than the required are available
            N_feat_to_be_added = N_feats -m;
            feat_M = all_images{obj,startPose,2};
    %          features(m+1:m+N_feat_to_be_added,:) = 0;
        else
            feat_M = all_images{obj,startPose,2}(1:N_feats,:);
        end
            
%         feat_M = features;
        %for loop doesn't execute if poses
        for i=(startPose+1):N_poses %concatenate all poses required
            [m,~] = size(all_images{obj,i,2});
            if m < N_feats %if fewer feats than the required are available
                N_feat_to_be_added = N_feats -m;
                features = all_images{obj,i,2};
    %             features(m+1:m+N_feat_to_be_added,:) = 0;
            else
                features = all_images{obj,i,2}(1:N_feats,:);
            end
            
            %concatenate them vertically
            feat_M = [feat_M; features];
        end % of for loop
    end
end
% % end