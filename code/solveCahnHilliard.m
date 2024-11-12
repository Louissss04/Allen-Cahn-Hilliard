function [L2_error, H1_error, is_stable] = solveCahnHilliard(node, elem, K, epsilon, dt, T, isPlot, is_g_nonzero)
    % solveCahnHilliard 求解 Cahn-Hilliard 方程并计算 L2 和 H1 误差
    %
    % 输入参数:
    %   node          - 节点坐标矩阵，大小为 N x 2
    %   elem          - 单元连接矩阵，大小为 NT x 3
    %   K             - 稳定项系数 (标量)
    %   epsilon       - ε 参数 (标量)
    %   dt            - 时间步长 (标量)
    %   T             - 总时间 (标量)
    %   isPlot        - 是否绘制结果 (1: 是, 0: 否)
    %   is_g_nonzero  - 源项是否为非零 (1: 非零, 0: 零)
    %
    % 输出参数:
    %   L2_error  - 每个时间步的 L2 误差向量，大小为 (numSteps + 1) x 1
    %   H1_error  - 每个时间步的 H1 误差向量，大小为 (numSteps + 1) x 1
    %   is_stable - 能量是否随时间递减的标志 (1: 是, 0: 否)

    %% 1. 参数和网格初始化

    % 重要常数
    N = size(node, 1);     % 节点数目
    NT = size(elem, 1);    % 单元数目
    Ndof = N;              % 自由度个数

    % 时间参数
    numSteps = round(T / dt); % 时间步数

    %% 2. 定义精确解及其梯度

    % 使用符号计算自动生成精确解的梯度
    syms x_sym y_sym t_sym;

    % 定义精确解 exact_u_sym（根据需要更改此表达式以进行不同的真解实验）
    % 示例真解：u(x, y, t) = exp(-t) * cos(pi*x) * cos(pi*y)
    exact_u_sym = exp(-t_sym) * cos(pi*x_sym) * cos(pi*y_sym);

    % 计算梯度 Du 的符号表达式
    du_dx_sym = diff(exact_u_sym, x_sym);
    du_dy_sym = diff(exact_u_sym, y_sym);

    % 将符号表达式转换为函数句柄
    exact_u_handle = matlabFunction(exact_u_sym, 'Vars', {x_sym, y_sym, t_sym});
    exact_du_dx_handle = matlabFunction(du_dx_sym, 'Vars', {x_sym, y_sym, t_sym});
    exact_du_dy_handle = matlabFunction(du_dy_sym, 'Vars', {x_sym, y_sym, t_sym});

    % 定义 Du 的函数句柄
    exact_Du = @(pxy, t) [ exact_du_dx_handle(pxy(:,1), pxy(:,2), t), ...
                           exact_du_dy_handle(pxy(:,1), pxy(:,2), t) ];

    % 设置 exact_u 为函数句柄
    exact_u = exact_u_handle;

    %% 3. 计算源项 g
    if is_g_nonzero
        g_func = compute_g(exact_u, epsilon);
    else
        g_func = @(x, y, t) zeros(size(x));
    end

    %% 4. 初始化存储变量
    u_all = zeros(N, numSteps + 1);        % 存储所有时间步的解，N x (numSteps + 1)
    u = exact_u(node(:,1), node(:,2), 0);  % 初始条件
    u_all(:, 1) = u;                        % 保存初始条件
    energy = zeros(numSteps + 1, 1);       % 存储能量
    L2_error = zeros(numSteps + 1, 1);     % 存储 L2 误差
    H1_error = zeros(numSteps + 1, 1);     % 存储 H1 误差

    %% 5. 计算基函数梯度和面积
    [Dphi, area] = gradbasis(node, elem); % 假设 gradbasis 返回 [NT x 2 x 3] 的 Dphi 和 [NT x 1] 的 area

    %% 6. 组装刚度矩阵 S
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

    %% 7. 组装质量矩阵 M
    M = sparse(Ndof, Ndof);
    for i = 1:3
        for j = i:3
            if i == j
                Mij = (1/6) * area; % 对角部分
            else
                Mij = (1/12) * area; % 非对角部分
            end
            M = M + sparse([elem(:, i); elem(:, j)], [elem(:, j); elem(:, i)], [Mij; Mij], Ndof, Ndof);
        end
    end

    %% 8. 定义能量函数
    F_energy = @(u) (1/4) * (u.^2 - 1).^2;

    %% 9. 获取积分点和权重
    [lambda, weight] = quadpts(4); % 假设 quadpts 返回 [nQuad x 3] 的 lambda 和 [nQuad x 1] 的 weight
    nQuad = size(lambda, 1);

    %% 10. 计算初始能量和误差
    energy_grad = 0;
    energy_potential = 0;
    for p = 1:nQuad
        % 计算积分点在全局坐标中的位置
        pxy = lambda(p, 1) * node(elem(:, 1), :) + ...
              lambda(p, 2) * node(elem(:, 2), :) + ...
              lambda(p, 3) * node(elem(:, 3), :);

        % 当前解在每个节点上的值
        u1 = u(elem(:, 1));
        u2 = u(elem(:, 2));
        u3 = u(elem(:, 3));

        % 插值获取 u 在积分点位置的值
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

    % 计算初始 L2 误差
    L2_error(1) = getL2error(node, elem, @(p) exact_u(p(:,1), p(:,2), 0), u);

    % 计算初始 H1 误差
    exact_Du_initial = @(pxy) exact_Du(pxy, 0);
    H1_error(1) = getH1error(node, elem, exact_Du_initial, u, K, 4); % quadOrder = 4

    %% 11. 时间步进循环
    for n = 1:numSteps
        % 当前时间
        current_t = n * dt;

        %% 11.1 计算源项 g
        if is_g_nonzero
            g_local = zeros(NT, 3);
            for p = 1:nQuad
                % 积分点在全局坐标中的位置
                pxy = lambda(p, 1) * node(elem(:, 1), :) + ...
                      lambda(p, 2) * node(elem(:, 2), :) + ...
                      lambda(p, 3) * node(elem(:, 3), :);

                % 计算 g_p
                g_p = g_func(pxy(:,1), pxy(:,2), current_t); % NT x 1

                % 累加每个基函数的贡献
                for i = 1:3
                    g_local(:, i) = g_local(:, i) + weight(p) * lambda(p, i) .* g_p;
                end
            end
            g_local = g_local .* area; % NT x 3
            g_global = accumarray(elem(:), g_local(:), [Ndof, 1]); % 全局右端项
        else
            g_global = zeros(Ndof, 1);
        end

        %% 11.2 计算非线性项 F(u^n)
        f = @(u) u.^3 - u; % 定义非线性函数
        F_local = zeros(NT, 3); % 初始化局部右端项
        u_vals = u_all(elem(:, 1:3), n); % NT x 3

        for p = 1:nQuad
            % 积分点在全局坐标中的位置
            pxy = lambda(p, 1) * node(elem(:, 1), :) + ...
                  lambda(p, 2) * node(elem(:, 2), :) + ...
                  lambda(p, 3) * node(elem(:, 3), :);

            % 插值获取 u_p
            u_p = lambda(p, 1) * u_all(elem(:,1), n) + ...
                  lambda(p, 2) * u_all(elem(:,2), n) + ...
                  lambda(p, 3) * u_all(elem(:,3), n);

            % 计算 f(u_p)
            fp = f(u_p); % NT x 1

            % 累加每个基函数的贡献
            for i = 1:3
                F_local(:, i) = F_local(:, i) + weight(p) * lambda(p, i) .* fp;
            end
        end

        % 累加到全局右端项
        F_local = F_local .* area; % NT x 3
        F = accumarray(elem(:), F_local(:), [Ndof, 1]); % 全局右端项

        %% 11.3 构造并求解线性系统
        A = [M/dt, S; (K / epsilon^2) * M + S, -M];
        b = [(M/dt) * u + g_global; (K / epsilon^2) * M * u - F / epsilon^2];

        % 求解线性系统
        sol = A \ b;

        % 更新 u 和 w
        u = sol(1:Ndof);
        % w = sol(Ndof+1:end);

        % 存储当前时间步的解
        u_all(:, n + 1) = u;

        %% 11.4 计算能量
        energy_grad = 0;
        energy_potential = 0;
        for p = 1:nQuad
            % 积分点在全局坐标中的位置
            pxy = lambda(p, 1) * node(elem(:, 1), :) + ...
                  lambda(p, 2) * node(elem(:, 2), :) + ...
                  lambda(p, 3) * node(elem(:, 3), :);

            % 当前解在每个节点上的值
            u1 = u(elem(:, 1));
            u2 = u(elem(:, 2));
            u3 = u(elem(:, 3));

            % 插值获取 u 在积分点位置的值
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

        %% 11.5 计算误差
        % 计算 L2 误差
        exact_u_t = @(p) exact_u(p(:,1), p(:,2), current_t);
        L2_error(n + 1) = getL2error(node, elem, exact_u_t, u);

        % 计算 H1 误差
        exact_Du_current = @(pxy) exact_Du(pxy, current_t);
        H1_error(n + 1) = getH1error(node, elem, exact_Du_current, u, K, 4); % quadOrder = 4
    end

    %% 12. 检查能量是否递减

    % 检查能量是否递减或保持不变
    is_stable = all(diff(energy) <= 0);

    %% 13. 绘制结果
    if isPlot
%         figure;
%         for n_plot = 1:1000:numSteps+1
%             trisurf(elem, node(:, 1), node(:, 2), u_all(:, n_plot));
%             shading interp;
%             view(3);
%             colorbar;
%             title(['Time step ', num2str(n_plot)]);
%             pause(0.1);
%         end

        % 绘制能量随时间变化
        figure;
        plot(0:numSteps, energy);
        xlabel('Time Step');
        ylabel('Energy');
        title('Energy Dissipation');
        grid on;

%         % 绘制 L2 误差随时间变化
%         figure;
%         plot(0:numSteps, L2_error);
%         xlabel('Time Step');
%         ylabel('L2 Error');
%         title('L2 Error over Time');
%         grid on;
% 
%         % 绘制 H1 误差随时间变化
%         figure;
%         plot(0:numSteps, H1_error);
%         xlabel('Time Step');
%         ylabel('H1 Error');
%         title('H1 Error over Time');
%         grid on;
    end
end
