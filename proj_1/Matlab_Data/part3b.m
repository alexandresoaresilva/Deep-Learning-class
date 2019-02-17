clc, clear, close all
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

N_poses = 0:1:20;
N_feats = [2, 4, 8, 16];
%% 5 features, 16 features
no_matched_points = cell.empty();
accuracy = 0;
train_accuracies = cell.empty();

    % for k=N_train

% ith row is no. of tyrain poses from 5 to 20 
% jth column is its related no of features used
 for i=1:length(N_poses)
     for j = 1:length(N_feats)
        [no_matched_points{i,j}, accuracy(i,j)] =...
            get_matches(all_images, N_poses(i), N_feats(j));
        
     end
 end
 
 accuracy = accuracy.*100;
 plot(N_poses, accuracy(:,1))
 hold on
 grid on
 plot(N_poses, accuracy(:,2))
 plot(N_poses, accuracy(:,3))
 plot(N_poses, accuracy(:,4))
 yticks(0:10:110);
 xticks(0:2:20);
 ylim([0 105]);
 xlim([0 20]);
 xlabel('Number of Training Images');
 ylabel('Accuracy(%%)');
 legend({'2 features';'4 features';'8 features';'16 features'},...
     'Location','southeast');
% (accuracy1 )
    
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
            no_matched_points(i,j) =...
                get_match_pts(all_images, i,j, N_pose, N_feat);
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

% feature vectors
function no_of_matched_feats =...
    get_match_pts(all_images, obj1,obj2, N_poses, N_feats)
    %train
%     [feat_M1, SURF_M_concat1] =...
%         get_feat_M_and_multi_SURF_points(all_images, obj1,...
%                                    1, N_poses, N_feats);
    feat_M1 = get_feat_M(all_images, obj1,1, N_poses, N_feats);

    feat_M2 = get_feat_M(all_images, obj2,(N_poses + 1), 24, N_feats);
    
    %test
%     [feat_M2, SURF_M_concat2] =...
%         get_feat_M_and_multi_SURF_points(all_images, obj2,...
%                                         (N_poses + 1), 24, N_feats);

    indexPairs = matchFeatures(feat_M1, feat_M2,'Unique',1);
    no_of_matched_feats = length(indexPairs);
%     matchedPoints1 = SURF_M_concat1(indexPairs(:,1));
%     matchedPoints2 = SURF_M_concat2(indexPairs(:,2));
end
 
% this takes care of smaller
function feat_M =...
    get_feat_M(all_images, obj, startPose, N_poses, N_feats)
    %gets no of features available
    
    
    if ~N_poses % in the case of 0 features. feat matrix is N_feats x 64 
        % filled up with zeros
        feat_M = single(zeros(N_feats,64));
    else
        [m,~] = size(all_images{obj,startPose,2});
        features  = 0;

        if m < N_feats %if fewer feats than the required are available
            N_feat_to_be_added = N_feats -m;
    %         all_images{obj,startPose,2};
             features = all_images{obj,startPose,2};
    %          features(m+1:m+N_feat_to_be_added,:) = 0;
        else
            features = all_images{obj,startPose,2}(1:N_feats,:);
        end

        feat_M = features;

        for i= (startPose+1):N_poses %concatenate all poses required

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
        end
    end
end
% 
% 
% % this takes care of smaller
% function feat_M =...
%     get_feat_M(all_images, obj, startPose, N_poses, N_feats)
%     %gets no of features available
%     [m,~] = size(all_images{obj,startPose,2});
%     features  = 0;
%     
%     if m < N_feats %if fewer feats than the required are available
%         N_feat_to_be_added = N_feats -m;
% %         all_images{obj,startPose,2};
%          features = all_images{obj,startPose,2};
% %          features(m+1:m+N_feat_to_be_added,:) = 0;
%     else
%         features = all_images{obj,startPose,2}(1:N_feats,:);
%     end
%     
%     feat_M = features;
%     
%     for i= (startPose+1):N_poses %concatenate all poses required
%         
%         [m,~] = size(all_images{obj,i,2});
%         if m < N_feats %if fewer feats than the required are available
%             N_feat_to_be_added = N_feats -m;
%             features = all_images{obj,i,2};
% %             features(m+1:m+N_feat_to_be_added,:) = 0;
%         else
%             features = all_images{obj,i,2}(1:N_feats,:);
%         end
%         %concatenate them vertically
%         feat_M = [feat_M; features];
%     end
% end

% 
% function [no_matched_points, accuracy] =...
%     get_matches(all_images, N_pose, N_feat)
% 
%     for i=1:10 % i is the train object 
%         for j=1:10 % j is the test object
%             [matchedPoints1, ~] = ...
%                 get_match_pts(all_images, i,j, N_pose, N_feat);
%             no_matched_points(i,j) = length(matchedPoints1.Scale);
%         end
%         % after all matches have been made
%         % if train object has more features matched with the right test
%         max_j = max(no_matched_points(i,:));
%         index_max = find(max_j == no_matched_points(i,:));
% 
%         if index_max == i
%             accuracy(i) = 1;
%         else
%             accuracy(i) = 0;
%         end
%     end
%     
%     train_accuracies{1,1} = no_matched_points;
%     train_accuracies{1,2} = accuracy;
% end

% 
% function [matchedPoints1, matchedPoints2] = ...
%     get_match_pts(all_images, obj1,obj2, N_poses, N_feats)
%     %train
% %     [feat_M1, SURF_M_concat1] =...
% %         get_feat_M_and_multi_SURF_points(all_images, obj1,...
% %                                    1, N_poses, N_feats);
%     feat_M1 = get_feat_M(all_images, obj1,1, N_poses, N_feats);
% 
%     feat_M2 = get_feat_M(all_images, obj2,(N_poses + 1), 24, N_feats);
%     
%     %test
% %     [feat_M2, SURF_M_concat2] =...
% %         get_feat_M_and_multi_SURF_points(all_images, obj2,...
% %                                         (N_poses + 1), 24, N_feats);
% 
%     indexPairs = matchFeatures(feat_M1, feat_M2,'Unique',1);
%     no_of_matched_feats = size(indexPairs);
% %     matchedPoints1 = SURF_M_concat1(indexPairs(:,1));
% %     matchedPoints2 = SURF_M_concat2(indexPairs(:,2));
% end
 

% % this takes care of smaller
% function [feat_M, SURF_M_concat] =...
%     get_feat_M_and_multi_SURF_points(all_images, obj,...
%                                     startPose, N_poses, N_feats)
%     %gets no of features available
%     [m,~] = size(all_images{obj,startPose,2});
%     features  = 0;
%     SURF_obj = 0;
%     
%     if m < N_feats %if fewer feats than the required are available
%         features = all_images{obj,startPose,2};
%         SURF_obj = all_images{obj,startPose,3};
%     else
%         features = all_images{obj,startPose,2}(1:N_feats,:);
%         SURF_obj = all_images{obj,startPose,3}(1:N_feats);
%     end
%     
%     feat_M = features;
%     SURF_M_concat = SURF_obj;
%     
%     for i= (startPose+1):N_poses %concatenate all poses required
%         
%         [m,~] = size(all_images{obj,i,2});
%         if m < N_feats %if fewer feats than the required are available
%             features = all_images{obj,i,2};
%             SURF_obj = all_images{obj,i,3};
%         else
%             features = all_images{obj,i,2}(1:N_feats,:);
%             SURF_obj = all_images{obj,i,3}(1:N_feats);
%         end
%         %concatenate them vertically
%         SURF_M_concat = vertcat(SURF_M_concat, SURF_obj);
%         feat_M = [feat_M; features];
%     end
% end