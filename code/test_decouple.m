clear; 

% 1. 定义初始网格大小和生成初始网格
h = 1/2;
[node, elem] = squaremesh([0, 1, 0, 1], h);

% 2. 定义模拟参数
dt = 1e-3;         
pde = mycoscosdata(dt);

% 3. 预分配误差数组
n = 4;                            % 网格细化
L2error = zeros(n + 1, 1);        % 存储各层级的 L2 误差
H1error = zeros(n + 1, 1);        % 存储各层级的 H1 误差
h_vals = zeros(n + 1, 1);         % 存储各层级的网格大小 h
h_vals(1) = h;

% 4. 运行初始网格并计算误差
[u, ~, L2error(1), H1error(1)] = SAVdecouple(node, elem, pde, dt);

% 5. 进行网格细化并计算误差
for m = 1:n
    % 细化网格
    [node, elem] = uniformrefine(node, elem);
    h_vals(m+1) = h_vals(m) / 2; % 网格大小减半

    % 调用求解函数并计算误差
    [~, ~, L2error(m+1), H1error(m+1)] = SAVdecouple(node, elem, pde, dt);
end


% 显示收敛阶表
fprintf('收敛阶:\n');
fprintf('Mesh Level\t h\t\t L2 Error\t\t H1 Error\t L2 Rate\t H1 Rate\n');
for m = 1:n+1
    if m == 1
        % 初始网格，没有收敛阶
        rate_L2 = '-';
        rate_H1 = '-';
    else
        % 计算收敛阶
        rate_L2 = log(L2error(m-1) / L2error(m)) / log(2);
        rate_H1 = log(H1error(m-1) / H1error(m)) / log(2);
        % 格式化为两位小数
        rate_L2 = sprintf('%.2f', rate_L2);
        rate_H1 = sprintf('%.2f', rate_H1);
    end
    fprintf('%d\t\t %.5f\t %.5e\t %.5e\t %s\t\t %s\n', ...
            m, h_vals(m), L2error(m), H1error(m), rate_L2, rate_H1);
end

% 绘制 L2 误差随网格大小的变化
figure;
loglog(h_vals, L2error, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
grid on;
xlabel('h', 'FontSize', 14);
ylabel('L2 Error', 'FontSize', 14);
title('L2 Error', 'FontSize', 16);
legend('L2 Error', 'Location', 'Best');
hold off;

% 绘制 H1 误差随网格大小的变化
figure;
loglog(h_vals, H1error, 'r-s', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
grid on;
xlabel('h', 'FontSize', 14);
ylabel('H1 Error', 'FontSize', 14);
title('H1 Error', 'FontSize', 16);
legend('H1 Error', 'Location', 'Best');
hold off;
