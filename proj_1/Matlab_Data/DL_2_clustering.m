%%
%Clustering
%kmeans cluster
clc, clear, close all
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
a = gcf;
a.WindowState = 'maximized';

k = 4; % number of classes
pbaspect([1 1 1])
km = kmeans(X,k, 'Distance','cityblock','Display','iter');
figure
subplot(221);
scatter3(X(:,1), X(:,3), X(:,4),10,km)
% legend(k_list);
axis vis3d
title(['Clustering 1: k-means, ',num2str(k), ' classes']);
% gscatter(A,B,km,...
%     [0,0.75,0.75;0.75,0,0.75;0.75,0.75,0],'...');
subplot(222);
silhouette(X,km,'cityblock');
title(['Clustering 1: silhouette k-means, ',num2str(k), ' classes']);
h = gca;
h.Children.EdgeColor = [.8 .8 1];
xlabel 'Silhouette Value'
ylabel 'Cluster'
pbaspect([1 1 1])
% daspect([1 1 1])

%Hierarchial 
Z = linkage(X,'ward');
% GMModel = fitgmdist(X,k);
% c = cluster(GMModel,X);
c = cluster(Z,'Maxclust',k);
subplot(223);
scatter3(X(:,1),X(:,3),X(:,4),10, c)
title(['Clustering 2: Gaussian Mixture Models, ',num2str(k), ' classes']);
% legend(k_list);
% title('Hierarchial');
axis vis3d

subplot(224);
silhouette(X,c,'cityblock');
title(['Clustering 2: silhouette Gaussian Mixture Models, ',num2str(k), ' classes']);
a = gcf;
a.WindowState = 'maximized';
% figure
% dendrogram(Z);