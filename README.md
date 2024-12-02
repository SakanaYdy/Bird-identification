# 数字图像处理大作业报告

作者：24210124 杨大宇

Github主页：https://github.com/SakanaYdy

项目仓库地址：https://github.com/SakanaYdy/Bird-identification

[TOC]

## 前置开发环境

Matlab 2024A

Pycharm

Python 3.11

注：不同的matlab编译器适配不同的python版本，如下图：

![img](https://pic.leetcode.cn/1733127545-zTqjDd-9ad038445f70b9a5f803e8c1951707ba.png)

## 整体页面演示



## 灰度直方图

### 灰度直方图概念

灰度直方图是关于灰度级分布的函数，是对图像中灰度级分布的统计。灰度直方图是将数字图像中的所有像素，按照**灰度值大小，统计其出现的频次**。灰度直方图是灰度级的函数，它表示图像中具有某种灰度级的像素的个数，反映了图像中某种灰度出现的频率。

### 灰度直方图显示实现

在这个公式中，image(y,x)表示图像在坐标 (y,x) 处的灰度值，$δimage(y,x)+1,k$ 是代数符号， 用于判断灰度值是否等于 k，如果等于则为1，否则为0。整个公式对图像的所有像素进行遍历求和，得到每个灰度级别的像素数量。简单来说就是**对灰度在图像内计数。**
$$
\text{{histogram}}(k) = \sum_{x=1}^{width} \sum_{y=1}^{height} \delta_{\text{{image}}(y, x) + 1, k}
$$

```matlab
        % 统计图像灰度直方图
        % 图像的输入是单通道的灰度图
        function histogram = gray_histogram(app, image)
            histogram = zeros(256, 1);
            [height, width] = size(image);

            for x = 1:width
                for y = 1:height
                    pixelValue = image(y, x) + 1;	% 0~255 映射到 1 ~ 256
                    histogram(pixelValue) = histogram(pixelValue) + 1;
                end
            end
        end
```

绘制灰度直方图数组到坐标轴

```matlab
        % 绘制灰度直方图数组到坐标轴
        function draw_histogram(app, histogram)
            % 获取灰度直方图坐标轴组件
            axesHandle = app.UIAxesGrayHistogramHistgram;
            % 绘制直方图
            bar(axesHandle, histogram);

        end
```

图片读取回调函数

计算灰度图像，图像控件赋值，调用绘制灰度图数组到坐标轴函数。

```matlab
        % Image clicked function: IImgHistogramInput
        function ImgHistogramInputOnClicked(app, event)
            [filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg, *.png, *.bmp)'}, '选择图像文件');
            if isequal(filename, 0) || isequal(pathname, 0)
                return;
            end
            fullFilePath = fullfile(pathname, filename);
            imgData = imread(fullFilePath);
            app.IImgHistogramInput.ImageSource = imgData;
            gray_image = rgb2gray(imgData);
            app.ImgHistogramGray.ImageSource =  cat(3, gray_image, gray_image, gray_image);
            histogram = gray_histogram(app, rgb2gray(app.IImgHistogramInput.ImageSource));
            draw_histogram(app, histogram);
        end
```

![image.png](https://pic.leetcode.cn/1703157471-XlYNwQ-image.png)

### 直方图均衡化

采用直方图均衡化，可以把原始图像的直方图变换为均匀分布（均衡）的形式，这样就增加了像素之间灰度值差别的动态范围，从而达到增强图像整体对比度的效果。

换言之，直方图均衡化的基本原理是：**对在图像中像素个数多的灰度值（即对画面起主要作用的灰度值）进行展宽，而对像素个数少的灰度值（即对画面不起主要作用的灰度值）进行归并**，从而**增大对比度，使图像清晰，达到增强的目的。**


```matlab
        % 直方图均衡化
        function equalized_image  = histogram_equalization(~, image)

            histgram = imhist(image);
            [h, w] = size(image);
            equalized_image = zeros(h, w, 'uint8'); % 初始化为uint8图像
            s = zeros(256);
            s(1) = histgram(1);

            for t = 2 : 256
                s(t) = s(t - 1) + histgram(t);
            end

            for x = 1: w
                for y = 1: h
                	% s数组是求前缀和，然后求向下取整进行平坦化
                    equalized_image(y, x) = uint8(255 * s(image(y, x) + 1) / (w * h));
                end
            end
        end
```

![image.png](https://pic.leetcode.cn/1703158474-vLGFfy-image.png)

灰度图均衡化前后对比：

![image.png](https://pic.leetcode.cn/1703158491-rHihyg-image.png)

![image.png](https://pic.leetcode.cn/1703158502-elFEUl-image.png)

### 总结

灰度直方图通过统计频次的方式显示图像的灰度分布信息。为了增强图像的对比度效果，直方图均衡化的方法，是通过将频次较多的灰度值进行展宽，出现频次较少的灰度值进行合并，从而达到增强对比度的效果。

## 图像增强

### 线性灰度增强

$$
g(x, y) = tan\alpha \cdot f(x, y)
$$

```matlab
        % 线性灰度级增强
        function new_image = linear_enhance(~, image, tan_alpha)
            % 对每个像素应用线性变换
            new_image = tan_alpha * double(image);
            new_image = uint8(max(0, min(255, new_image)));
        end
```

### 非线性对数增强

由于对数曲线在像素值较低的区域斜率大，在像素值较高的区域斜率较小，所以图像经过对数变换后，**较暗区域的对比度将有所提升，**所以就可以增强图像的暗部细节。

对数变换可以将图像的低灰度值部分扩展，显示出低灰度部分更多的细节，将其高灰度值部分压缩，减少高灰度值部分的细节，从而达到强调图像低灰度部分的目的。变换方法：
$$
g(x, y) = c \cdot log [f(x, y) + 1]
$$

```matlab
 		% 非线性对数变换
        function new_image = log_enhance(~, image, c)
            new_image = c * log(1 + double(image));
            new_image = uint8(max(0, min(255, new_image)));
        end
```

### 非线性指数增强

指数变换的作用是扩展图像的高灰度级、压缩低灰度级。虽然幂次变换也有这个功能，但是图像经过指数变换后对比度更高，高灰度级也被扩展到了更宽的范围。
$$
g(x, y) = b ^ {  c \cdot [f(x, y) - \alpha]  } - 1
$$

```matlab
        % 非线性指数变换
        function new_image = exp_enhance(~, image, a, b, c)
            new_image = power(b, c * (image - a)) - 1;
            new_image = uint8(max(0, min(255, new_image)));
        end
```



![image.png](https://pic.leetcode.cn/1703161174-MXbCuS-image.png)

可视化控件中，使用了微调器Spinner获取各类变换的参数值。

```matlab
        % Value changed function: SpinnerEnhanceTan
        function SpinnerEnhanceTanValueChanged(app, event)
            value = app.SpinnerEnhanceTan.Value;
            new_image = linear_enhance(app, rgb2gray(app.ImgEnhanceGray.ImageSource), value);
            app.ImgEnhanceLinear.ImageSource = cat(3, new_image, new_image, new_image);
        end

        % Value changed function: SpinnerEnhanceLog
        function SpinnerEnhanceLogValueChanged(app, event)
            value = app.SpinnerEnhanceLog.Value;
            new_image = log_enhance(app, rgb2gray(app.ImgEnhanceGray.ImageSource), value);
            app.ImgEnhanceLog.ImageSource = cat(3, new_image, new_image, new_image);
        end

        % Button pushed function: BtnEnhanceExp
        function BtnEnhanceExpButtonPushed(app, event)
            a = app.SpinnerEnhanceExpa.Value;
            b = app.SpinnerEnhanceExpb.Value;
            c = app.SpinnerEnhanceExpc.Value;
            new_image = exp_enhance(app, rgb2gray(app.ImgEnhanceGray.ImageSource), a, b, c);
            app.ImgEnhanceExp.ImageSource = cat(3, new_image, new_image, new_image);
        end
```

### 总结

在图像灰度增强中，线性增强通过对元素值乘上增强系数，实现提亮的效果；而对数增强，由于对数函数在低灰度值部分的斜率较大，在高灰度值部分的斜率较小，因而可以实现对暗部提亮的暗部增强效果；而指数函数通过调整参数，可以实现展宽图像的高灰度级，压缩图像的低灰度级。可以看到参数的调整可以实现非常好的对比度效果。

## 图像旋转缩放

### 双线性插值

双线性插值是一种在离散的数据点之间估算数值的插值方法，其主要应用之一是图像缩放。算法的基本思想是利用目标像素周围的四个邻近像素的值，通过线性插值的方式计算目标像素的值。分别是$ (x,y),(x+1,y),(x,y+1),(x+1,y+1)(x,y),(x+1,y),(x,y+1),(x+1,y+1)$
$$
I'(x', y') = (1 - a)(1 - b)I(x, y) + a(1 - b)I(x + 1, y) + (1 - a)bI(x, y + 1) + abI(x + 1, y + 1)
$$

其中，

$$
x = \frac{x'}{k_x}, \quad y = \frac{y'}{k_y}
$$

```matlab
 % 图像缩放
        function new_image = scale(~, image, k_x, k_y)
            [h, w, ~] = size(image);
            new_h = round(h * k_y); % 更改画布大小
            new_w = round(w * k_x);

            new_image = zeros(new_h, new_w, 3);  % Initialize the output image with 3 channels for RGB.

            for x_ = 1:new_w
                for y_ = 1:new_h
                    x = x_ / k_x;
                    y = y_ / k_y;
                    a = x - floor(x);
                    b = y - floor(y);
                    x = floor(x);
                    y = floor(y);

                    x = max(1, min(w - 1, x));
                    y = max(1, min(h - 1, y));

                    f_x_y_b = image(y, x, :) + b * (image(y + 1, x, :) - image(y, x, :));
                    f_x_1_y_b = image(y, x + 1, :) + b * (image(y + 1, x + 1, :) - image(y, x + 1, :));
                    f_x_a_y_b = f_x_y_b + a * (f_x_1_y_b - f_x_y_b);
                    new_image(y_, x_, :) = f_x_a_y_b;
                end
            end

            new_image = uint8(new_image);  % 转换为 uint8 类型以便正确显示
        end
```

图像旋转采用逆变换思想+双线性插值。

新的图像大小由原始图像的宽度 w 和高度 h 以及旋转角度 θ 决定。
$$
new_h=round(h⋅∣cos(θ)∣+w⋅∣sin(θ)∣) \\
new_w=round(h⋅∣sin⁡(θ)∣+w⋅∣cos⁡(θ)∣) \\
$$
逆变换公式为：

$(x,y)(x,y) 是旋转后图像中的像素坐标，(new_x,new_y)(new_x,new_y) 是对应于原始图像中的坐标。$
$$
new_x=cos⁡(θ)⋅(x−center_x)−sin⁡(θ)⋅(y−center_y)+center_x
\\
new_y=sin⁡(θ)⋅(x−center_x)+cos⁡(θ)⋅(y−center_y)+center_y
$$


```matlab
 % 图像旋转
        function rotated_image = rotate(~, image, theta)
            [h, w, ~] = size(image);

            % 计算旋转后的图像大小，round四舍五入确保是整数
            new_h = round(h * abs(cosd(theta)) + w * abs(sind(theta)));
            new_w = round(h * abs(sind(theta)) + w * abs(cosd(theta)));

            rotated_image = zeros(new_h, new_w, 3);

            % 计算旋转中心
            center_x = w / 2;
            center_y = h / 2;

            for x_ = 1:new_w
                for y_ = 1:new_h
                    x = x_ - center_x;
                    y = y_ - center_y;

                    % 根据逆变换公式，计算旋转后的坐标
                    % 这是在原图中的位置
                    new_x = cosd(theta) * x - sind(theta) * y + center_x;
                    new_y = sind(theta) * x + cosd(theta) * y + center_y;

                    % 使用双线性插值估算颜色值
                    a = new_x - floor(new_x);
                    b = new_y - floor(new_y);
                    x1 = max(1, min(w - 1, floor(new_x)));
                    y1 = max(1, min(h - 1, floor(new_y)));
                    x2 = min(x1 + 1, w);
                    y2 = min(y1 + 1, h);

                    f_x_y_b = image(y1, x1, :) + b * (image(y2, x1, :) - image(y1, x1, :));
                    f_x_1_y_b = image(y1, x2, :) + b * (image(y2, x2, :) - image(y1, x2, :));
                    f_x_a_y_b = f_x_y_b + a * (f_x_1_y_b - f_x_y_b);
                    rotated_image(y_, x_, :) = f_x_a_y_b;
                end
            end

            rotated_image = uint8(rotated_image);
        end

```



![image.png](https://pic.leetcode.cn/1703162154-PdUBeG-image.png)

使用Slider输入缩放倍数：

![image.png](https://pic.leetcode.cn/1703162950-kooALd-image.png)

使用Knob组件，输入旋转角度：

![image.png](https://pic.leetcode.cn/1703163011-cACxDy-image.png)

## 加噪与去噪

### 加噪

椒盐噪声

椒盐噪声(salt-and-pepper noise)又称脉冲噪声，它随机改变一些像素值，在二值图像上表现为使一些像素点变白，一些像素点变黑。 是由图像传感器，传输信道，解码处理等产生的黑白相间的亮暗点噪声，也就是老人们比较熟悉的所谓“雪花”。

```matlab
        function SliderNFsaltDValueChanged(app, event)
            value = app.SliderNFsaltD.Value;
            app.ImgNFnoise.ImageSource =  imnoise(app.ImgNFInput.ImageSource, 'salt & pepper', value);
        end
```

高斯噪声

```matlab
        % Button pushed function: BtnNFg
        function BtnNFgButtonPushed(app, event)
            m = app.SliderNFgm.Value;    % 均值
            Var = app.SpinnerNFgVar.Value;   % 方差
            app.ImgNFnoise.ImageSource = imnoise(app.ImgNFInput.ImageSource,'gaussian',m, Var);
        end
```

### 去噪

#### 空间域滤波

由图像像元组成的空间。在图像空间中以长度(距离)为自变量直接对像元值进行处理称为空间域处理。



#### 频率域滤波

以频率(即波数)为自变量描述图像的特征,可以将一幅图像像元值在空间上的变化分解为具有不同振幅、空间频率和相位的简振函数的线性叠加,图像中各种频率成分的组成和分布称为空间频谱。这种对图像的频率特征进行分解、处理和分析称为频率域处理或波数域处理。



#### 双边滤波

双边滤波与传统的高斯滤波不同，双边滤波考虑了像素之间的空间距离和像素值之间的颜色差异。这使得它能够在平滑图像的同时保留图像的细节和边缘信息。
$$
I'(x, y) = \frac{1}{W_p} \sum_{(i, j) \in \Omega} I(i, j) \cdot w(i, j, x, y)
$$

$$
Ω 表示像素 (i,j)(i,j) 的邻域，通常是一个窗口或者一个卷积核。

W_p 是归一化权重，确保滤波后的像素值在 0 到 1 之间。
$$



其中：
$$
w(i, j, x, y) = w_s(i, j, x, y) \cdot w_r(i, j, x, y)
$$

空间域权重：空间距离越近，权重越大

$$
w_s(i, j, x, y) = e^{-\frac{(i - x)^2 + (j - y)^2}{2 \sigma_s^2}}
$$

像素值权重：像素值越相近，权重越大

$$
w_r(i, j, x, y) = e^{-\frac{(I(i, j) - I(x, y))^2}{2 \sigma_r^2}}
$$


```matlab
 % 双边滤波
        function filter_image = filter_double(~, image, w, sigma_s, sigma_r)
            image = im2double(image);

            [X, Y] = meshgrid(-w:w, -w:w);

            % 计算邻域内的空间权值
            Gs = exp(-(X.^2 + Y.^2) / (2 * sigma_s^2));

            dim = size(image);
            filter_image = zeros(dim);

            for i = 1:dim(1)
                for j = 1:dim(2)
                    % 确定作用区域
                    iMin = max(i - w, 1);
                    iMax = min(i + w, dim(1));
                    jMin = max(j - w, 1);
                    jMax = min(j + w, dim(2));

                    I = image(iMin:iMax, jMin:jMax, :); % 提取该区域的源图像值赋给 I
                    % 计算灰度临近权值
                    Gr = exp(-(sum((I - image(i, j, :)).^2, 3)) / (2 * sigma_r^2));

                    % W 为Gr 和 G的乘积
                    W = Gr .* Gs((iMin:iMax) - i + w + 1, (jMin:jMax) - j + w + 1);
                    filter_image(i, j, :) = sum(sum(W .* I, 1), 2) / sum(W(:));
                end
            end
        end
```



基于模糊技术的加权均值滤波

```matlab
 % 基于模糊技术的加权均值滤波
        function filter_image = filter_mohu(~, image, N)
            % N = N;
            [m, n] = size(image);
            image = double(image);
            filter_image = image;
            % 计算权重的分母
            sNum = N^2 - 1;

            % 计算卷积核的半径
            N = (N - 1) / 2;

            % 遍历图像的像素
            for i = 1 + N : m - N
                for j = 1 + N : n - N
                    sum = 0;

                    % 计算局部像素差的平方和
                    for m = i - N : i + N
                        for n = j - N : j + N
                            d = (image(i, j) - image(m, n))^2;
                            sum = sum + d;
                        end
                    end

                    % 计算模糊参数 beta
                    beta = sum / sNum;
                    sum2 = 0;
                    sum3 = 0;

                    % 应用加权均值滤波
                    for m = i - N : i + N
                        for n = j - N : j + N
                            d = (image(i, j) - image(m, n))^2;
                            % 计算权重
                            mu = exp(-d / beta);
                            % 加权均值的分子
                            sum2 = sum2 + mu / beta * image(m, n);
                            % 加权均值的分母
                            sum3 = sum3 + mu / beta;
                        end
                    end

                    % 更新结果图像的像素值
                    filter_image(i, j) = sum2 / sum3;
                end
            end

            % 将结果图像转换为8位无符号整数型
            filter_image = uint8(filter_image);
        end
```

![image.png](https://pic.leetcode.cn/1703163636-KAhVJq-image.png)

### 总结

在滤波中，双边滤波通过在高斯滤波的基础上，考虑到像素值相近程度的权重影响，再加上空间距离的权重影响，实现了图像的滤波。但如果参数调整过大，虽然平滑度更高，但是图像的模糊程度也更高。

基于模糊技术的平滑滤波，在计算时间复杂度上过高，但是当$n$ 达到7以上的时候，效果比较好。

## 边缘检测

图像边缘是图像最基本的特征，所谓**边缘**(Edge) 是指图像局部特性的不连续性。灰度或结构, 颜色等信息的突变处称之为**边缘**。例如，灰度级的突变、颜色的突变、纹理结构的突变等。边缘是一个区域的结束，也是另一个区域的开始，利用该特征可以分割图像。

### 一阶导数的边缘算子

通过模板作为核与图像的每个像素点做卷积和运算，然后选取合适的阈值来提取图像的边缘。常见的有Roberts算子、Sobel算子和Prewitt算子。

### 二阶导数的边缘算子

依据于二阶导数过零点，常见的有Laplacian 算子，此类算子对噪声敏感。

### Roberts算子

**Roberts算子**又称为交叉微分算法，它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。常用来处理具有陡峭的低噪声图像，当图像边缘接近于正45度或负45度时，该算法处理效果更理想。其缺点是对边缘的定位不太准确，提取的边缘线条较粗。

```matlab
        function new_image = edge_roberts(~, f, T)

            [m, n] = size(f);
            new_image = zeros(m, n);

            for i = 2:m-1
                for j = 2:n-1
                    new_image(i,j)=abs(f(i+1,j+1)-f(i,j))+abs(f(i,j+1)-f(i+1,j));
                    if new_image(i,j)<T
                        new_image(i,j)=0;
                    else
                        new_image(i,j)=255;
                    end
                end
            end
            new_image = uint8(new_image);
        end
```



### Sobel算子

```matlab
 function new_image = edge_sobel(~, f, T)
            [m, n] = size(f);
            new_image = zeros(m, n);

            for i = 2:m-1
                for j = 2:n-1
                    new_image(i,j)=abs(f(i-1,j-1)+2*f(i,j-1)+f(i+1,j-1)-f(i-1,j+1)-2*f(i,j+1)-f(i+1,j+1))+abs(f(i+1,j-1)+2*f(i+1,j)+f(i+1,j+1)-f(i-1,j-1)-2*f(i-1,j)-f(i-1,j+1));
                    if new_image(i, j) < T
                        new_image(i, j) = 0;
                    else
                        new_image(i, j) = 255;
                    end
                end
            end
            new_image = uint8(new_image);
        end

```



### Prewitt算子

```matlab
function new_image = edge_prewitt(~, f, T)
            [m, n] = size(f);
            new_image = zeros(m, n);

            for i = 2:m-1
                for j = 2:n-1

                    new_image(i,j)=abs(f(i-1,j-1)+f(i,j-1)+f(i+1,j-1)-f(i-1,j+1)-f(i,j+1)-f(i+1,j+1))+abs(f(i+1,j-1)+f(i+1,j)+f(i+1,j+1)-f(i-1,j-1)-f(i-1,j)-f(i-1,j+1));

                    if new_image(i, j) < T
                        new_image(i, j) = 0;
                    else
                        new_image(i, j) = 255;
                    end
                end
            end
            new_image = uint8(new_image);
        end
```

### 拉普拉斯算子

```matlab
function new_image = edge_laplacian(~, f, T)
            [m, n] = size(f);
            new_image = zeros(m, n);

            for i = 2:m-1
                for j = 2:n-1
                    new_image(i,j) = 4 * f(i, j) - f(i-1, j) - f(i+1, j) - f(i, j-1) - f(i, j+1);

                    if new_image(i, j) < T
                        new_image(i, j) = 0;
                    else
                        new_image(i, j) = 255;
                    end
                end
            end
            new_image = uint8(new_image);
        end

```



![image.png](https://pic.leetcode.cn/1703165310-JpLpjv-image.png)

![image.png](https://pic.leetcode.cn/1703165402-OoJprV-image.png)

### 总结

Robert它是基于交叉差分的梯度算法，通过局部差分计算检测边缘线条。常用来处理具有陡峭的低噪声图像，当图像边缘接近于正45度或负45度时，该算法处理效果更理想。其缺点是对边缘的定位不太准确，提取的边缘线条较粗。

而拉普拉斯是二阶微分算子，而且对于噪声敏感，在本问题中，效果十分不好，无法准确提取出边缘。

## 图像分割

### Otsu最大类间方差法

将图像的像素分为两个类别：前景和背景。具体步骤如下：

1. 统计图像的灰度直方图，得到每个灰度级的像素数目。
2. 遍历所有可能的阈值（0到255），计算根据该阈值将图像分为前景和背景的类内方差。
3. 根据类内方差的最小值确定最佳阈值。

该算法的优点是简单易懂，计算效率高。它适用于灰度图像二值化处理，特别是对于具有双峰直方图的图像效果更好。然而，该算法对于具有非双峰直方图的图像可能产生较差的分割结果。因此，在应用大津算法之前，需要对图像的直方图进行分析，确保适用性。

```matlab
        function BtnFGOstuButtonPushed(app, event)
            % 转换为灰度图像
            imgData = app.ImgFGinput.ImageSource;
            grayImage = rgb2gray(imgData);

            % 使用Ostu最大类间方差法，进行阈值分割（可根据实际情况调整阈值）
            threshold = graythresh(grayImage);
            
            
            app.LabelFGOstu.Text = threshold * 255 + "";
            binaryImage = imbinarize(grayImage, threshold);
            % 对二值图像进行后处理（去除小的区域）
            binaryImage = bwareaopen(binaryImage, 1000);
            % 创建彩色蒙版图像
            if app.FGreverse == false
                colorMask = ~cat(3, binaryImage, binaryImage, binaryImage);
            else
                colorMask = cat(3, binaryImage, binaryImage, binaryImage);
            end
            % 将彩色蒙版叠加到原始图片上
            segmentedResult = imgData;
            % 获取满足条件的像素索引
            indices = colorMask;
            % 将这些像素设置为白色
            segmentedResult(indices) = 255;
            binaryImage = cat(3, binaryImage, binaryImage, binaryImage);
            binaryImage = uint8(binaryImage) * 255;
            app.ImgFGbin.ImageSource = binaryImage;
            app.ImgFGres.ImageSource = segmentedResult;

        end
```



![image.png](https://pic.leetcode.cn/1703166712-uRfGXT-image.png)

### Kmeans聚类

将图像转换到HSV控件，使用 k-means 算法对图像的颜色进行聚类，然后通过生成二进制目标掩码来标识聚类的结果。

```matlab
        function targetMask = clusterImage(~, image, K)

            hsv = rgb2hsv(image);
            h = hsv(:, :, 1);
            h(h > 330 / 360) = 0;		% 阈值剔除
            training = h(:);
            startdata = linspace(0, 1, K)';
            [IDX, ~] = kmeans(training, K, 'Start', startdata);	% kmeans聚类
            targetMask = reshape(IDX == 1, size(h));

        end
```

![image.png](https://pic.leetcode.cn/1703166762-dVzbAr-image.png)

### 区域生长法

区域生长可以根据预先定义的生长规则将像素或者小区域不断组合为更大区域的过程。具体地，区域生长是从一组初始种子点出发，通过预先定义的区域生长规则，将与种子点性质相似的领域像素不断添加到每个种子点上，并且满足区域生长的终止条件时形成最终生长区域的过程。

这里的“性质相似”是灰度差距在一定阈值内为标准，灰度差超过某个值为生长终止条件。

实现了交互式的选择某个区域作为种子，使用imfreehand函数，绘制不规则初始种子。

```matlab
        function segmentedMask = interactiveRegionGrowing(app, image)
            grayImage = rgb2gray(image);
            figure;
            imshow(image);
            title('交互式选择种子);
            h = imfreehand;
            mask = createMask(h);
            segmentedMask = regionGrowing(app, grayImage, mask);
        end
        function segmentedMask = regionGrowing(app, image, seedMask)
            threshold = app.KnobFGseed.Value; 
            segmentedMask = zeros(size(image));
            [rows, cols] = find(seedMask);
            seedPoints = [rows, cols];
            for i = 1:size(seedPoints, 1)
                currentSeed = seedPoints(i, :);
                if segmentedMask(currentSeed(1), currentSeed(2)) == 0
                    segmentedMask = growRegion(app, image, segmentedMask, currentSeed, threshold);
                end
            end
        end
```



![image.png](https://pic.leetcode.cn/1703166859-ttwTeD-image.png)

### 主动轮廓法（额外寻找的方法）

由于前面的阈值分割，不能非常全面考虑到边缘和颜色的问题，而区域生长也都是受限于灰度条件。

而前景的叶片具有比较鲜明的轮廓，所以拓展了主动轮廓法。使用drawpolygon允许用户初始化绘制一个轮廓，然后通过不断向内部迭代逼近轮廓边缘。最终效果与迭代次数有关。

```matlab
 % Button pushed function: Btnactive
        function BtnactiveButtonPushed(app, event)
            imgData = app.ImgFGinput.ImageSource;
            I = rgb2gray(app.ImgFGinput.ImageSource);
            imshow(I);
            title('Original Image');

            r =  drawpolygon ;
            mask = createMask(r);

            imshow(mask)
            title('Initial Contour Location')

            binaryImage = activecontour(I,mask,1500,'edge');
            % 将逻辑图像转换为RGB图像
            rgbImage = cat(3, binaryImage, binaryImage, binaryImage);
            % 将RGB图像转换为uint8类型
            rgbImage = uint8(rgbImage * 255);

            app.ImgFGbin.ImageSource = rgbImage;

            % 创建彩色蒙版图像
            if app.FGreverse == false
                colorMask = ~cat(3, binaryImage, binaryImage, binaryImage);
            else
                colorMask = cat(3, binaryImage, binaryImage, binaryImage);
            end

            segmentedResult = imgData;

            % 获取满足条件的像素索引
            indices = colorMask;
            % 将这些像素设置为白色

            indices(end) = 1;
            segmentedResult(indices) = 255;

            binaryImage = cat(3, binaryImage, binaryImage, binaryImage);

            binaryImage = uint8(binaryImage) * 255;

            app.ImgFGres.ImageSource = segmentedResult;


        end
```



![image.png](https://pic.leetcode.cn/1703166913-tHFaNZ-image.png)

可以看到最终比较有效的分割出了叶片。

### 总结

本问题中，对于叶片前景从背景中提取出来的难点在于，叶片本身的灰度值跨度也比较大，而且有些部分可能跟背景的灰度比较接近。

因而，在图像分割中，使用的阈值分割都无法很好的将前景和背景进行分割分离。比如Ostu最大类间方差法，是通过对灰度进行二分类的枚举，寻找两个类间方差最大的一种分割。但是可以看到，叶片的暗部和图像右下角的暗部被分割在了一起，而叶片的亮部反而没有分割出来。

同理，其他的手动阈值选择也无法很准确的分割。而区域生长法，生长结束条件也是基于阈值差的，所以也会出现右下角的暗部也被分割进去的不理想情况。

因而，我在后续过程中，分析叶片的特征，它有比较统一的颜色（绿色），和比较明确的边缘。所以首先考虑基于颜色HSV空间进行Kmeans聚类，但是可以看到，叶子上的蓝色斑点并没有分割进去。

所以最终采用了基于迭代的主动轮跨方法，需要交互式的给定一个初始边缘，最终能够在迭代1500次的时候逼近叶片边缘轮廓。



## 深度学习分类模型

### VGG 模型



### EfficientNet 模型
