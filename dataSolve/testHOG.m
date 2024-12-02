% 直接执行HOG特征提取测试

% 读取图像
img = imread('1.jpg');  % 替换为你自己的图片路径
imgData = rgb2gray(img); % 如果是彩色图像，先转换为灰度图像

% 将图像数据转换为双精度类型，方便计算
imgData = double(imgData);

% 设置HOG参数
cell_size = 8;  % cell的大小
block_size = 2; % block的大小
bins = 9;       % 方向直方图的bin数

% 调用HOG特征提取函数
[hog_features, hog_image] = hog_feature(imgData, cell_size, block_size, bins);

% 显示HOG特征直方图
disp('HOG特征：');
disp(hog_features);

% 可视化HOG特征图像
figure;
imshow(hog_image, []);
title('HOG特征图像');


% 计算HOG特征
function [hog_features, hog_image] = hog_feature(image, cell_size, block_size, bins)

    % 将输入图像转换为双精度
    image = double(image);
    
    % 计算梯度（水平和垂直方向）
    [Gx, Gy] = imgradientxy(image);
    
    % 计算每个像素的梯度幅度和方向
    [mag, ang] = imgradient(Gx, Gy);
    
    % 将图像分成小块（cell）
    [height, width] = size(image);
    num_cells_x = floor(width / cell_size);
    num_cells_y = floor(height / cell_size);
    
    % 初始化HOG图像和特征向量
    hog_image = zeros(height, width);
    hog_features = [];
    
    % 计算每个cell的梯度直方图
    for i = 1:num_cells_y
        for j = 1:num_cells_x
            % 获取cell区域
            cell_x_start = (j-1) * cell_size + 1;
            cell_x_end = j * cell_size;
            cell_y_start = (i-1) * cell_size + 1;
            cell_y_end = i * cell_size;
            
            % 获取cell内的梯度方向和幅度
            cell_mag = mag(cell_y_start:cell_y_end, cell_x_start:cell_x_end);
            cell_ang = ang(cell_y_start:cell_y_end, cell_x_start:cell_x_end);
            
            % 计算方向梯度直方图
            hog_hist = compute_histogram(cell_mag, cell_ang, bins);
            
            % 将每个cell的HOG特征存储到特征向量中
            hog_features = [hog_features, hog_hist];
            
            % 可视化：构建HOG图像
            for x = 1:cell_size
                for y = 1:cell_size
                    angle = cell_ang(y, x);
                    magnitude = cell_mag(y, x);
                    % 将梯度可视化为箭头
                    hog_image(cell_y_start + y - 1, cell_x_start + x - 1) = magnitude;
                end
            end
        end
    end
    
    % 归一化HOG特征
    hog_features = normalize_hog(hog_features, block_size, num_cells_x, num_cells_y);

end

% 计算方向梯度直方图
function hist = compute_histogram(mag, ang, bins)
    % 将角度转换到0-180度区间
    ang = mod(ang, 180);
    
    % 创建方向梯度直方图
    hist = zeros(1, bins);
    
    % 计算每个像素的梯度方向对应的bin
    for i = 1:numel(ang)
        bin = floor(ang(i) / (180 / bins)) + 1;
        hist(bin) = hist(bin) + mag(i);
    end
end

% 归一化HOG特征
function hog_features = normalize_hog(hog_features, block_size, num_cells_x, num_cells_y)
    % 归一化块（block）
    num_blocks_x = num_cells_x - block_size + 1;
    num_blocks_y = num_cells_y - block_size + 1;
    normalized_hog_features = [];
    
    for i = 1:num_blocks_y
        for j = 1:num_blocks_x
            block_start_idx = (i-1) * num_cells_x + j;
            block_end_idx = block_start_idx + block_size * block_size - 1;
            block = hog_features(block_start_idx:block_end_idx);
            
            % L2 归一化
            norm_factor = norm(block, 2);
            if norm_factor > 0
                block = block / norm_factor;
            end
            
            normalized_hog_features = [normalized_hog_features, block];
        end
    end
    
    hog_features = normalized_hog_features;
end
