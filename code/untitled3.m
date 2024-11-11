clear;

% 初始网格大小
h(1) = 1/2;

% 生成初始网格
[node, elem] = squaremesh([0, 1, 0, 1], h(1));

% 定义参数
epsilon = 1;
K = 5;
dt = 1e-6;
T = 2000*dt;       
isPlot = 0;

% 预分配误差数组
n = 5; % 网格细化
L2error = zeros(n + 1, 1);
H1error = zeros(n + 1, 1);
h_vals = zeros(n + 1, 1);
h_vals(1) = h(1);

% 运行初始网格
[L2_error, H1_error] = solveCahnHilliard(node, elem, K, epsilon, dt, T, isPlot);
L2error(1) = L2_error(end);
H1error(1) = H1_error(end);

% 进行网格细化并计算误差
for m = 1:n
    % 细化网格
    [node, elem] = uniformrefine(node, elem);
    h_vals(m+1) = h_vals(m) / 2; % 网格大小减半

    % 调用求解函数
    [L2_error, H1_error] = solveCahnHilliard(node, elem, K, epsilon, dt, T, isPlot);
    
    % 存储最终时间步的误差
    L2error(m+1) = L2_error(end);
    H1error(m+1) = H1_error(end);
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

% 计算并显示收敛阶
fprintf('Convergence Rates:\n');
fprintf('Mesh Level\t h\t\t L2 Rate\t H1 Rate\n');
for m = 1:n
    rate_L2 = log(L2error(m) / L2error(m+1)) / log(2);
    rate_H1 = log(H1error(m) / H1error(m+1)) / log(2);
    fprintf('%d\t\t %.5f\t %.2f\t\t %.2f\n', m, h_vals(m), rate_L2, rate_H1);
end
