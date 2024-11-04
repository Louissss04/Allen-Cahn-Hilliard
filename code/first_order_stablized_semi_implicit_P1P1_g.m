clear;

% Cahn-Hilliard 方程一阶稳定半隐式 P1-P1 元 

% Mesh generation
[node, elem] = squaremesh([0, 2*pi, 0, 2*pi], pi/4);
showmesh(node, elem);

% Important constants
N = size(node, 1); % 节点数目
NT = size(elem, 1); % 单元数目
Ndof = N; % 自由度个数
epsilon = 0.5; % epsilon 参数
K = 0.5; % 稳定项系数 K 

% Time
dt = 1e-6; % 时间步长
T = 0.1; % 总时间
numSteps = round(T / dt); % 时间步数

% Init
u_all = zeros(N, numSteps + 1); % 存储所有时间步的解，N x (numSteps + 1)
u = 0.05 * sin(node(:, 1)) .* sin(node(:, 2)); % 初始条件 u(x,y,0) = 0.05 * sin(x) * sin(y)
u_all(:, 1) = u; % 保存初始条件
energy = zeros(numSteps + 1, 1);
L2_error = zeros(numSteps + 1, 1); % 存储每个时间步的 L2 误差

% 定义精确解和源项 g 的匿名函数
exact_u = @(x, y, t) 0.05 * exp(-t) .* sin(x) .* sin(y);
g_func = @(x, y, t) (0.15 - 0.10 / epsilon^2) * exp(-t) .* sin(x) .* sin(y) ...
    - (0.00075 / epsilon^2) * exp(-3 * t) .* (sin(x) .* cos(x).^2 .* sin(y).^3 + ...
    sin(y) .* cos(y).^2 .* sin(x).^3 - sin(x).^3 .* sin(y).^3);

% Dphi = Dlambda
[Dphi, area] = gradbasis(node, elem);

% Stiff matrix S
S = sparse(Ndof, Ndof);
for i = 1:3
    for j = i:3
        Sij = (Dphi(:, 1, i) .* Dphi(:, 1, j) + Dphi(:, 2, i) .* Dphi(:, 2, j)) .* area;
        if j == i
            S = S + sparse(elem(:, i), elem(:, j), Sij, Ndof, Ndof);
        else
            S = S + sparse([elem(:, i); elem(:, j)], [elem(:, j); elem(:, i)], [Sij; Sij], Ndof, Ndof);
        end
    end
end

% Mass matrix M
M = sparse(Ndof, Ndof);
for i = 1:3
    for j = i:3
        Mij = (1/12) * area; % 线性基函数的质量矩阵部分
        if i == j
            Mij = Mij * 2; % 对角部分
        end
        M = M + sparse([elem(:, i); elem(:, j)], [elem(:, j); elem(:, i)], [Mij; Mij], Ndof, Ndof);
    end
end

% Energy function
F_energy = @(u) (1/4) * (u.^2 - 1).^2;
energy_grad = 0;
energy_potential = 0;
[lambda, weight] = quadpts(4); % 获取积分点和权重
nQuad = size(lambda, 1);

% 初始能量计算
for p = 1:nQuad
    % 计算积分点在全局坐标中的位置
    pxy = lambda(p, 1) * node(elem(:, 1), :) ...
        + lambda(p, 2) * node(elem(:, 2), :) ...
        + lambda(p, 3) * node(elem(:, 3), :);

    % 获取当前时间步解 u 在每个节点上的值
    u1 = u(elem(:, 1));
    u2 = u(elem(:, 2));
    u3 = u(elem(:, 3));

    % 通过插值获取 u 在积分点位置的值
    u_p = lambda(p, 1) * u1 + lambda(p, 2) * u2 + lambda(p, 3) * u3;

    % 计算非线性势能项 F(u_p)
    fp = F_energy(u_p);

    % 计算每个单元的梯度 grad_u
    grad_u = Dphi(:, :, 1) .* u1 + Dphi(:, :, 2) .* u2 + Dphi(:, :, 3) .* u3; % NT x 2

    % 计算 |grad_u|^2
    grad_u_sq = grad_u(:, 1).^2 + grad_u(:, 2).^2; % NT x 1

    % 累加梯度能量项
    energy_grad = energy_grad + weight(p) * (epsilon^2 / 2) * grad_u_sq .* area;

    % 累加势能项
    energy_potential = energy_potential + weight(p) * fp .* area;
end

energy(1) = sum(energy_grad) + sum(energy_potential);
L2_error(1) = getL2error(node, elem, @(p) exact_u(p(:,1), p(:,2), 0), u); % 初始 L2 误差

% 开始模拟
for n = 1:numSteps
    % 计算源项 g
    current_t = (n+1) * dt;
    
    % 在积分点处计算 g
    g_local = zeros(NT, 3);
    for p = 1:nQuad
        % 计算积分点在全局坐标中的位置
        pxy = lambda(p, 1) * node(elem(:, 1), :) ...
            + lambda(p, 2) * node(elem(:, 2), :) ...
            + lambda(p, 3) * node(elem(:, 3), :);
        
        % 通过匿名函数计算 g_p
        g_p = g_func(pxy(:,1), pxy(:,2), current_t); % NT x 1
            
        % 累加每个基函数的贡献
        for i = 1:3
            g_local(:, i) = g_local(:, i) + weight(p) * lambda(p, i) .* g_p;
        end
    end
    g_local = g_local .* area; % NT x 3
    g_global = accumarray(elem(:), g_local(:), [Ndof, 1]); % 累加得到全局右端项

    % 计算非线性项 F(u^n)
    f = @(u) u.^3 - u; % 定义非线性函数 f(u)
    F_local = zeros(NT, 3); % 初始化局部右端项
    u_vals = u_all(elem(:, 1:3), n); % NT x 3

    for p = 1:nQuad
        % 计算积分点在全局坐标中的位置
        pxy = lambda(p, 1) * node(elem(:, 1), :) ...
            + lambda(p, 2) * node(elem(:, 2), :) ...
            + lambda(p, 3) * node(elem(:, 3), :);

        % 通过匿名函数获取 u_p
        u_p = lambda(p, 1) * u_all(elem(:,1), n) + lambda(p, 2) * u_all(elem(:,2), n) + lambda(p, 3) * u_all(elem(:,3), n);

        % 计算非线性项 f(u_p) 的值
        fp = f(u_p); % NT x 1

        % 累加每个基函数在当前积分点的贡献
        for i = 1:3
            F_local(:, i) = F_local(:, i) + weight(p) * lambda(p, i) .* fp;
        end
    end

    % 考虑每个单元的面积并累加到全局右端项
    F_local = F_local .* area; % NT x 3
    F = accumarray(elem(:), F_local(:), [Ndof, 1]); % 累加得到全局右端项

    % 构造整体矩阵和右端向量以一次性求解 u^{n+1} 和 w^{n+1}
    % 构造左端矩阵 A 和右端向量 b
    A = [M/dt, S; (K / epsilon^2) * M + S, -M];
    b = [(M/dt) * u + g_global; (K / epsilon^2) * M * u - F / epsilon^2];

    % 求解线性系统 A * [u^{n+1}; w^{n+1}] = b
    sol = A \ b;

    % 更新 u 和 w
    u = sol(1:Ndof);
    w = sol(Ndof+1:end);

    % 存储当前时间步的解
    u_all(:, n + 1) = u;

    % 计算当前时间步的能量
    energy_grad = 0;
    energy_potential = 0;
    for p = 1:nQuad
        % 计算积分点在全局坐标中的位置
        pxy = lambda(p, 1) * node(elem(:, 1), :) ...
            + lambda(p, 2) * node(elem(:, 2), :) ...
            + lambda(p, 3) * node(elem(:, 3), :);

        % 获取当前时间步解 u 在每个节点上的值
        u1 = u(elem(:, 1));
        u2 = u(elem(:, 2));
        u3 = u(elem(:, 3));

        % 通过插值获取 u 在积分点位置的值
        u_p = lambda(p, 1) * u1 + lambda(p, 2) * u2 + lambda(p, 3) * u3;

        % 计算非线性势能项 F(u_p)
        fp = F_energy(u_p);

        % 计算每个单元的梯度 grad_u
        grad_u = Dphi(:, :, 1) .* u1 + Dphi(:, :, 2) .* u2 + Dphi(:, :, 3) .* u3; % NT x 2

        % 计算 |grad_u|^2
        grad_u_sq = grad_u(:, 1).^2 + grad_u(:, 2).^2; % NT x 1

        % 累加梯度能量项
        energy_grad = energy_grad + weight(p) * (epsilon^2 / 2) * grad_u_sq .* area;

        % 累加势能项
        energy_potential = energy_potential + weight(p) * fp .* area;
    end
    energy(n + 1) = sum(energy_grad) + sum(energy_potential);

    % 计算 L2 误差
    exact_u_t = @(p) exact_u(p(:,1), p(:,2), current_t);
    L2_error(n + 1) = getL2error(node, elem, exact_u_t, u);
end

% plot
figure;
for n = 1:1000:numSteps+1
    trisurf(elem, node(:, 1), node(:, 2), u_all(:, n));
    shading interp;
    view(3);
    colorbar;
    title(['Time step ', num2str(n)]);
    pause(0.1);
end

% 能量随时间变化
figure;
plot(0:10*dt:T, energy(1:10:end));
xlabel('Time');
ylabel('Energy');
title('Energy Dissipation');
grid on;

% L2 误差随时间变化
figure;
plot(0:10*dt:T, L2_error(1:10:end));
xlabel('Time');
ylabel('L2 Error');
title('L2 Error over Time');
grid on;
