% 在 MATLAB 中加载模块
py.importlib.import_module('vgg');

% 调用函数
% result = py.efficientNet.add(1, 2);
% disp(result);


res = py.vgg.predict("1.jpg");
disp(res);
% py.test.add(1,2)

% py.efficientNet.add(1,2)