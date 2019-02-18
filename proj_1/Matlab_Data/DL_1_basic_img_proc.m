clc, clear, close all
%% a. onion grayscale
I_gray = rgb2gray(imread('onion.png'));
imwrite(I_gray,'onion_gray.png');
subplot(131)
imshow(I_gray);
title('a. onion grayscale');
%% b. Intro to the 2D Fourier space
I = imread('cameraman.tif');
If = fftshift(fft2(I));
If_log = log(1 + abs(If));
If_disp = If_log/max(If_log(:));
subplot(132)
imshow(I,[]);
title('b.1. Intro to the 2D Fourier space');
subplot(133)
imshow(If_disp,[]);
title('b.2. Intro to the 2D Fourier space');
a = gcf;
a.WindowState = 'maximized';
% ====================================
% >>> b.3. low pass filter
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
title('b.3. low pass filter');
a = gcf;
a.WindowState = 'maximized';
% ====================================
% >>> b.4. high pass filter
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
title('b.4. high pass filter');
a = gcf;
a.WindowState = 'maximized';
%% c. Spacial filtering and convolution
filt =@(n)1/n^2*ones(n);

I = imread('coins.png');
v = [1 2 1];
sobel_x = [-v' [0 0 0]' v'];
sobel_y = [-v; 0 0 0; v];
int_filt = filt(5);
dI_dx = conv2(I, sobel_x);
dI_dy = conv2(I, sobel_y);
intI = conv2(I, int_filt);

figure
subplot(131);
imshow(I, []);
title('c. filtering and convolution: orig. image');

subplot(132);
imshow(dI_dx + dI_dy, []);
title({'c. convolution dI_dx + dI_dy'; 'non-ideal high pass filter'});
% title('b.3. low pass filter');
subplot(133);
imshow(intI, []);
title({'c. convolution integrating filter'; 'non-ideal low pass filter'});
a = gcf;
a.WindowState = 'maximized';