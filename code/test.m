clear; 
clc;

% 1. 定义初始网格大小和生成初始网格
h = 1/4;
[node, elem] = squaremesh([0, 1, 0, 1], h);

% 2. 定义模拟参数
epsilon = 0.01;       % ε 参数
K = 1;           % 稳定项系数
dt = 1e-3;         % 时间步长
T = 20*dt;         % 总时间
isPlot = 1;        % 是否绘制结果
is_g_nonzero = 0;  % 是否使用非零源项（1: 非零, 0: 零）

% 3. 预分配误差数组
n = 4;                            % 进行4次网格细化
L2error = zeros(n + 1, 1);        % 存储各层级的 L2 误差
H1error = zeros(n + 1, 1);        % 存储各层级的 H1 误差
h_vals = zeros(n + 1, 1);         % 存储各层级的网格大小 h
h_vals(1) = h;

% 预分配稳定性数组
is_stable_all = zeros(n + 1, 1);   % 存储各层级的稳定性标志

% 4. 运行初始网格并计算误差
[L2_error, H1_error, is_stable] = solveCahnHilliard(node, elem, K, epsilon, dt, T, isPlot, is_g_nonzero);
L2error(1) = L2_error(end);
H1error(1) = H1_error(end);
is_stable_all(1) = is_stable;

% 5. 进行网格细化并计算误差
for m = 1:n
    % 细化网格
    [node, elem] = uniformrefine(node, elem);
    h_vals(m+1) = h_vals(m) / 2; % 网格大小减半

    % 调用求解函数并计算误差
    [L2_error, H1_error, is_stable] = solveCahnHilliard(node, elem, K, epsilon, dt, T, isPlot, is_g_nonzero);
    
    % 存储最终时间步的误差
    L2error(m+1) = L2_error(end);
    H1error(m+1) = H1_error(end);
    is_stable_all(m+1) = is_stable;
end

% 显示当前模拟参数
fprintf('当前模拟参数:\n');
fprintf('K = %.4f\n', K);
fprintf('epsilon = %.4f\n', epsilon);
fprintf('dt = %.4e\n', dt);
fprintf('T = %.4e\n\n', T);

% 6. 根据 is_g_nonzero 输出不同信息
if is_g_nonzero
    % 显示收敛阶表
    fprintf('收敛阶:\n');
    fprintf('Mesh Level\t h\t\t L2 Error\t\t H1 Error\t L2 Rate\t H1 Rate\n');
    for m = 1:n
        rate_L2 = log(L2error(m) / L2error(m+1)) / log(2);
        rate_H1 = log(H1error(m) / H1error(m+1)) / log(2);
        fprintf('%d\t\t %.5f\t %.5e\t %.5e\t %.2f\t\t %.2f\n', ...
                m, h_vals(m), L2error(m), H1error(m), rate_L2, rate_H1);
    end

    % 绘制 L2 误差随网格大小的变化
    figure;
    loglog(h_vals, L2error, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
    hold on;
    grid on;
    xlabel('h', 'FontSize', 14);
    ylabel('L2 Error', 'FontSize', 14);
    title('L2 Error vs Mesh Size', 'FontSize', 16);
    legend('L2 Error', 'Location', 'Best');
    hold off;

    % 绘制 H1 误差随网格大小的变化
    figure;
    loglog(h_vals, H1error, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
    hold on;
    grid on;
    xlabel('h', 'FontSize', 14);
    ylabel('H1 Error', 'FontSize', 14);
    title('H1 Error vs Mesh Size', 'FontSize', 16);
    legend('H1 Error', 'Location', 'Best');
    hold off;
else
    % 当 is_g_nonzero=0 时，输出每个网格层级的稳定性信息
    fprintf('稳定性结果:\n');
    fprintf('Mesh Level\t h\t\t Stable\n');
    for m = 1:n+1
        if is_stable_all(m)
            stability = 'Yes';
        else
            stability = 'No';
        end
        fprintf('%d\t\t %.5f\t %s\n', m, h_vals(m), stability);
    end
end
