clear;                   
clc;                      

% 1. 定义固定参数
h = 1/8;                 % 网格大小
epsilon = 0.001;            
isPlot = 0;               % 是否绘制结果（1: 是, 0: 否）
is_g_nonzero = 0;         % 是否使用非零源项（1: 非零, 0: 零）

% 2. 生成初始网格
[node, elem] = squaremesh([0, 1, 0, 1], h);

% 3. 定义变化的参数
% dt_values = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2];    % 时间步长数组
% K_values = [0, 0.5, 1];                  % 稳定项系数数组
% dt_values = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2];
% K_values = [0, 0.5, 0.65, 1];
dt_values = [2e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2];
K_values = [0, 0.5, 1];

% 4. 初始化稳定性结果表
num_K = length(K_values);                % K 的数量
num_dt = length(dt_values);              % dt 的数量
stability_table = strings(num_K, num_dt);% 初始化一个字符串数组存储稳定性结果

% 5. 循环遍历 K 和 dt，运行模拟并记录稳定性
for i = 1:num_K
    K = K_values(i);                      % 当前 K 值
    for j = 1:num_dt
        dt = dt_values(j);                % 当前 dt 值
        T = 20 * dt;                       % 总时间
        
        fprintf('运行模拟: K = %.2f, dt = %.1e, T = %.1e\n', K, dt, T);
        
        % 调用求解函数
        [~, ~, is_stable] = solveCahnHilliard(node, elem, K, epsilon, dt, T, isPlot, is_g_nonzero);
        
        % 记录稳定性
        if is_stable
            stability_table(i,j) = 'Yes';
        else
            stability_table(i,j) = 'No';
        end
    end
end

% 6. 输出稳定性表格
fprintf('\n稳定性结果(epsilon = %.3f):\n', epsilon);
fprintf('K \\ dt\t');

for j = 1:num_dt
    fprintf('%.1e\t', dt_values(j));
end
fprintf('\n');

for i = 1:num_K
    fprintf('%.2f\t', K_values(i));
    for j = 1:num_dt
        fprintf('%s\t\t', stability_table(i,j));
    end
    fprintf('\n');
end