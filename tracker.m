%% Part I
clear all, close all

% Load in mask image
target8b = imread('swain_database/crunchberries.sqr.128.bmp');
% Preprocess to eliminate background
for i = 1:size(target8b,1)
    for j = 1:size(target8b,2)
        if target8b(i,j,1) == 0 & target8b(i,j,2) == 0 & target8b(i,j,3) == 0
            target8b(i,j,1) = NaN;
            target8b(i,j,2) = NaN;
            target8b(i,j,3) = NaN;
        end
    end
end
% Convert each channel from 8-bit to 3-bit integers
target3b = fix(double(target8b)/(2^8/2^3))+1;

% Initialize colour histogram
M = zeros(8,8,8);
% Count occurance of each
for i = 1:size(target3b,1)
    for j = 1:size(target3b,2)
        currpixel = target3b(i,j,:);
        M(currpixel(1), currpixel(2), currpixel(3)) = M(currpixel(1), currpixel(2), currpixel(3)) + 1;
    end
end

% Load in image test
frame8b = imread('SwainCollageForBackprojectionTesting.bmp');
% Preprocess to eliminate background
for i = 1:size(frame8b,1)
    for j = 1:size(frame8b,2)
        if frame8b(i,j,1) == 0 & frame8b(i,j,2) == 0 & frame8b(i,j,3) == 0
            frame8b(i,j,1) = NaN;
            frame8b(i,j,2) = NaN;
            frame8b(i,j,3) = NaN;
        end
    end
end
% Convert each channel from 8-bit to 3-bit integers
frame3b = fix(double(frame8b)/(2^8/2^3))+1;

% Initialize colour histogram
I = zeros(8,8,8);
% Count occurance of each
for i = 1:size(frame3b,1)
    for j = 1:size(frame3b,2)
        currpixel = frame3b(i,j,:);
        I(currpixel(1), currpixel(2), currpixel(3)) = I(currpixel(1), currpixel(2), currpixel(3)) + 1;
    end
end


% Initialize colour histogram
R = zeros(8,8,8);
% Count occurance of each
for i = 1:size(frame3b,1)
    for j = 1:size(frame3b,2)
        currpixel = frame3b(i,j,:);
        R(currpixel(1), currpixel(2), currpixel(3)) = min(M(currpixel(1), currpixel(2), currpixel(3))/I(currpixel(1), currpixel(2), currpixel(3)),1);
    end
end


% Backproject ratio values onto image
for i = 1:size(frame3b,1)
    for j = 1:size(frame3b,2)
        currpixel = frame3b(i,j,:);
        backprojection(i,j,:) = R(currpixel(1),currpixel(2),currpixel(3));
    end
end

% Create disk mask
r = min(size(target3b,1),size(target3b,2))/2;
D = zeros(r*2);
for i = 1:size(D,1)
    for j = 1:size(D,2)
        if sqrt((size(D,1)/2 - i)^2 + (size(D,2)/2 - j)^2) < r
            D(i,j) = 1;
        end
    end
end

% Convolve disk mask with ratio image
backprojection = conv2(D, backprojection);

% Locate peak
[max_val, index] = max(backprojection(:));
backprojection = backprojection./max_val;

x = index/size(backprojection,1)-r;
y = mod(index,size(backprojection,1))-r;

% Show image with centroid
figure, imshow(frame8b), hold;
plot(x,y,'x','LineWidth',2,'Color','g'); % Highlight centroid
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
plot(x+xp,y+yp,'LineWidth',2,'Color','g');

% % Test intersection
% intersection = 0;
% normalizer = 0;
% for i = 1:8
%     for j = 1:8
%         for k = 1:8
%             intersection = intersection + min( I(i,j,k), M(i,j,k) );
%             normalizer = normalizer + M(i,j,k);
%         end
%     end
% end
% intersection/normalizer

%% Part II
clear all, close all

% Load in video file get input for first frame
video = load('CMPT412_blackcup.mat');
video = video.blackcup;
imshow(video(:,:,:,1));
[x_input,y_input] = ginput(1);
r = 50;

% Load in mask image
target8b = imcrop(video(:,:,:,1),[x_input-r y_input-r 2*r 2*r]);
% Preprocess to eliminate background
for i = 1:size(target8b,1)
    for j = 1:size(target8b,2)
        if target8b(i,j,1) == 0 & target8b(i,j,2) == 0 & target8b(i,j,3) == 0
            target8b(i,j,1) = NaN;
            target8b(i,j,2) = NaN;
            target8b(i,j,3) = NaN;
        end
    end
end
% Convert each channel from 8-bit to 3-bit integers
target3b = fix(double(target8b)/(2^8/2^3))+1;

% Initialize colour histogram
M = zeros(8,8,8);
% Count occurance of each
for i = 1:size(target3b,1)
    for j = 1:size(target3b,2)
        currpixel = target3b(i,j,:);
        M(currpixel(1), currpixel(2), currpixel(3)) = M(currpixel(1), currpixel(2), currpixel(3)) + 1;
    end
end

% Create disk mask
D = zeros(r*2);
for i = 1:size(D,1)
    for j = 1:size(D,2)
        if sqrt((size(D,1)/2 - i)^2 + (size(D,2)/2 - j)^2) < r
            D(i,j) = 1;
        end
    end
end

for frame = 1:size(video,4);
    % Close last open window
    close;
    % Load in image test
    frame8b = video(:,:,:,frame);
    % Convert each channel from 8-bit to 3-bit integers
    frame3b = fix(double(frame8b)/(2^8/2^3))+1;

    % Initialize colour histogram
    I = zeros(8,8,8);
    % Count occurance of each
    for i = 1:size(frame3b,1)
        for j = 1:size(frame3b,2)
            currpixel = frame3b(i,j,:);
            I(currpixel(1), currpixel(2), currpixel(3)) = I(currpixel(1), currpixel(2), currpixel(3)) + 1;
        end
    end

    % Initialize colour histogram
    R = zeros(8,8,8);
    % Count occurance of each
    for i = 1:size(frame3b,1)
        for j = 1:size(frame3b,2)
            currpixel = frame3b(i,j,:);
            R(currpixel(1), currpixel(2), currpixel(3)) = min(M(currpixel(1), currpixel(2), currpixel(3))/I(currpixel(1), currpixel(2), currpixel(3)),1);
        end
    end

    % Backproject ratio values onto image
    for i = 1:size(frame3b,1)
        for j = 1:size(frame3b,2)
            currpixel = frame3b(i,j,:);
            backprojection(i,j,:) = R(currpixel(1),currpixel(2),currpixel(3));
        end
    end

    % Convolve disk mask with ratio image
    backprojection = conv2(D, backprojection);

    % Locate peak
    [max_val, index] = max(backprojection(:));
    backprojection = backprojection./max_val;

    x_coord = index/size(backprojection,1)-r;
    y_coord = mod(index,size(backprojection,1))-r;

    % Show image with centroid
    figure, imshow(frame8b), hold;
    title(['Frame ' num2str(frame)]),
    plot(x_coord,y_coord,'x','LineWidth',2,'Color','g'); % Highlight centroid
    plot(x_coord+r*cos(0:0.01:2*pi),y_coord+r*sin(0:0.01:2*pi),'LineWidth',2,'Color','g');
    %figure, imshow(backprojection);
        
    filename = fullfile('tracking.gif');
    drawnow
    gifframe = getframe(1);
    im = frame2im(gifframe);
    [imind,cm] = rgb2ind(im,256);
    if frame == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
    clear backprojection;
    clear index;
    clear x_coord;
    clear y_coord;
end


%% Part III
clear all, close all

% Load in video file get input for first frame
video = load('CMPT412_bluecup.mat');
video = video.bluecup;
imshow(video(:,:,:,1));
[x_input,y_input] = ginput(1);
r = 50;

% Load in mask image
target8b = imcrop(video(:,:,:,1),[x_input-r y_input-r 2*r 2*r]);
% Preprocess to eliminate background
for i = 1:size(target8b,1)
    for j = 1:size(target8b,2)
        if target8b(i,j,1) == 0 & target8b(i,j,2) == 0 & target8b(i,j,3) == 0
            target8b(i,j,1) = NaN;
            target8b(i,j,2) = NaN;
            target8b(i,j,3) = NaN;
        end
    end
end
% Convert each channel from 8-bit to 3-bit integers
target3b = fix(double(target8b)/(2^8/2^3))+1;

% Initialize colour histogram
M = zeros(8,8,8);
% Count occurance of each
for i = 1:size(target3b,1)
    for j = 1:size(target3b,2)
        currpixel = target3b(i,j,:);
        M(currpixel(1), currpixel(2), currpixel(3)) = M(currpixel(1), currpixel(2), currpixel(3)) + 1;
    end
end

% Create disk mask
D = zeros(r*2);
for i = 1:size(D,1)
    for j = 1:size(D,2)
        if sqrt((size(D,1)/2 - i)^2 + (size(D,2)/2 - j)^2) < r
            D(i,j) = 1;
        end
    end
end

% Initialize mean as input
mean = [x_input y_input];

for frame = 1:size(video,4);
    % Close last open window
    close;
    % Load in image test
    frame8b = video(:,:,:,frame);
    % Convert each channel from 8-bit to 3-bit integers
    frame3b = fix(double(frame8b)/(2^8/2^3))+1;

    % Initialize colour histogram
    I = zeros(8,8,8);
    % Count occurance of each
    for i = 1:size(frame3b,1)
        for j = 1:size(frame3b,2)
            currpixel = frame3b(i,j,:);
            I(currpixel(1), currpixel(2), currpixel(3)) = I(currpixel(1), currpixel(2), currpixel(3)) + 1;
        end
    end

    % Initialize colour histogram
    R = zeros(8,8,8);
    % Count occurance of each
    for i = 1:size(frame3b,1)
        for j = 1:size(frame3b,2)
            currpixel = frame3b(i,j,:);
            R(currpixel(1), currpixel(2), currpixel(3)) = min(M(currpixel(1), currpixel(2), currpixel(3))/I(currpixel(1), currpixel(2), currpixel(3)),1);
        end
    end

    % Backproject ratio values onto image
    for i = 1:size(frame3b,1)
        for j = 1:size(frame3b,2)
            currpixel = frame3b(i,j,:);
            backprojection(i,j,:) = R(currpixel(1),currpixel(2),currpixel(3));
        end
    end

    % Convolve disk mask with ratio image
    backprojection = conv2(D, backprojection);

    % Calculate new mean
    total = 0;
    x_total = 0;
    y_total = 0;
    for i = 1:size(backprojection,1)
        for j = 1:size(backprojection,2)
            if sqrt((j-mean(1))^2 + (i-mean(2))^2) < 2*r
                total = total + backprojection(i,j);
                x_total = x_total + j*backprojection(i,j);
                y_total = y_total + i*backprojection(i,j);
            end
        end
    end
    mean(1) = x_total/total;
    mean(2) = y_total/total;

    % Show image with centroid
    figure, imshow(frame8b), hold;
    title(['Frame ' num2str(frame)]),
    plot(mean(1),mean(2),'x','LineWidth',2,'Color','g'); % Highlight centroid
    plot(mean(1)+r*cos(0:0.01:2*pi),mean(2)+r*sin(0:0.01:2*pi),'LineWidth',2,'Color','g');
        
    filename = fullfile('trackingmean.gif');
    drawnow
    gifframe = getframe(1);
    im = frame2im(gifframe);
    [imind,cm] = rgb2ind(im,256);
    if frame == 1;
      imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
    else
      imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
    clear backprojection;
    clear index;
    clear x_coord;
    clear y_coord;
end