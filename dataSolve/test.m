% 主脚本部分

% 读取图像
image = imread('1.jpg');  % 替换为你的图片路径

% 如果是彩色图像，转换为灰度图像
if size(image, 3) == 3
    image = rgb2gray(image);  % 转换为灰度图像
end

% 将图像数据转换为双精度类型，方便计算
image = double(image);

% 设置LBP参数
radius = 1;      % 半径
neighbors = 8;    % 邻居数量

% 调用LBP特征提取函数
[histogram, lbp_image] = lbp_histogram(image, radius, neighbors);

% 显示结果
disp('LBP直方图：');
disp(histogram);

% 可视化图像
figure;
imshow(image, []);
title('输入图像');

% 显示LBP模式图像
figure;
imshow(lbp_image, []);
title('LBP模式图像');

% 以下是函数定义部分：

function [histogram, lbp_image] = lbp_histogram(image, radius, neighbors)
    % 计算LBP特征直方图
    [height, width] = size(image);
    lbp_image = zeros(height, width);  % 初始化LBP图像
    histogram = zeros(2^neighbors, 1);  % LBP模式的总数是 2^neighbors

    for x = 1 + radius : width - radius
        for y = 1 + radius : height - radius
            % 获取邻域像素
            neighborhood = getNeighborhood(image, x, y, radius, neighbors);
            % 计算LBP值
            lbp_value = calculateLBP(neighborhood);
            % 计算直方图
            lbp_image(y, x) = lbp_value;
            histogram(lbp_value + 1) = histogram(lbp_value + 1) + 1;
        end
    end
end

function neighborhood = getNeighborhood(image, x, y, radius, neighbors)
    % 获取邻域的像素值
    angle_step = 2 * pi / neighbors;
    neighborhood = zeros(1, neighbors);
    
    for i = 1:neighbors
        angle = angle_step * (i - 1);
        offset_x = round(radius * cos(angle));
        offset_y = round(radius * sin(angle));
        neighborhood(i) = image(y + offset_y, x + offset_x);
    end
end

function lbp_value = calculateLBP(neighborhood)
    % 计算LBP值，返回一个整数值
    center_pixel = neighborhood(1);
    lbp_value = 0;

    for i = 2:length(neighborhood)
        if neighborhood(i) >= center_pixel
            lbp_value = lbp_value + 2^(i - 2);
        end
    end
end
