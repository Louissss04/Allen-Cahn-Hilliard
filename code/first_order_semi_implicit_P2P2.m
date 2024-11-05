clear;

% Cahn-Hilliard 方程一阶半隐式 P2-P2 元 

% Mesh generation
[node, elem] = squaremesh([0, 2*pi, 0, 2*pi], 0.25*pi);
showmesh(node, elem);

% Important constants
[elem2dof, edge, bdDof] = dofP2(elem);
N = size(node, 1); % 节点数目
NT = size(elem, 1); % 单元数目
NE = size(edge, 1); % 边数目
Ndof = N + NE; % 自由度个数
epsilon = 1; % epsilon 参数

% Time
dt = 1e-5; % 时间步长
T = 1; % 总时间
numSteps = round(T / dt); % 时间步数

% Init
u_all = zeros(Ndof, numSteps + 1); % 存储所有时间步的解，Ndof x (numSteps + 1)
u = 0.05 * sin(node(:, 1)) .* sin(node(:, 2)); % 初始条件 u(x,y,0) = 0.05 * sin(x) * sin(y)
u_all(1:N, 1) = u; % 将节点处的初始条件存储到对应自由度

% 插值计算边自由度处的初始条件
for e = 1:NE
    edgeNodes = edge(e, :);
    midPoint = 0.5 * (node(edgeNodes(1), :) + node(edgeNodes(2), :));
    u_all(N + e, 1) = 0.05 * sin(midPoint(1)) * sin(midPoint(2)); % 在边的中点坐标处计算初始条件
end

energy = zeros(numSteps + 1, 1); % 初始自由能

% 定义精确解和源项 g 的匿名函数
exact_u = @(x, y, t) 0.05 * exp(-t) .* sin(x) .* sin(y);
g_func = @(x, y, t) (0.15 - 0.10 / epsilon^2) * exp(-t) .* sin(x) .* sin(y) ...
    - (0.00075 / epsilon^2) * exp(-3 * t) .* (sin(x) .* cos(x).^2 .* sin(y).^3 + ...
    sin(y) .* cos(y).^2 .* sin(x).^3 - sin(x).^3 .* sin(y).^3);

% Dphi = Dlambda
[Dlambda, area] = gradbasis(node, elem); % 计算基函数梯度和单元面积
area = repmat(area, 1, 6); % 将面积扩展为与每个基函数对应的大小

% Get quadrature points and weights
[lambda, weight] = quadpts(3); 
nQuad = size(lambda, 1);

% Generate sparse pattern for stiffness and mass matrices
ii = zeros(21 * NT, 1); 
jj = zeros(21 * NT, 1); 
index = 0;
for i = 1:6
    for j = i:6
        ii(index+1:index+NT) = double(elem2dof(:, i));
        jj(index+1:index+NT) = double(elem2dof(:, j));
        index = index + NT;
    end
end

% Compute Dphi at quadrature points
Dphip = zeros(NT, 2, 6, nQuad);
for p = 1:nQuad
    Dphip(:,:,6,p) = 4 * (lambda(p,1) * Dlambda(:,:,2) + lambda(p,2) * Dlambda(:,:,1));
    Dphip(:,:,1,p) = (4 * lambda(p,1) - 1) .* Dlambda(:,:,1);
    Dphip(:,:,2,p) = (4 * lambda(p,2) - 1) .* Dlambda(:,:,2);
    Dphip(:,:,3,p) = (4 * lambda(p,3) - 1) .* Dlambda(:,:,3);
    Dphip(:,:,4,p) = 4 * (lambda(p,2) * Dlambda(:,:,3) + lambda(p,3) * Dlambda(:,:,2));
    Dphip(:,:,5,p) = 4 * (lambda(p,3) * Dlambda(:,:,1) + lambda(p,1) * Dlambda(:,:,3));
end

% Assemble stiffness matrix S for P2 elements
sA = zeros(21*NT, nQuad);
for p = 1:nQuad
    index = 0;
    for i = 1:6
        for j = i:6
            Aij = weight(p) * dot(Dphip(:,:,i,p), Dphip(:,:,j,p), 2);
            Aij = Aij .* area(:, i);
            sA(index+1:index+NT, p) = Aij;
            index = index + NT;
        end
    end
end
sA = sum(sA, 2);

% Create the sparse stiffness matrix
S = sparse(ii, jj, sA, Ndof, Ndof);

% Define shape functions at quadrature points
phip = zeros(nQuad, 6);
for p = 1:nQuad
    phip(p,6) = 4 * lambda(p,1) .* lambda(p,2);
    phip(p,1) = lambda(p,1) .* (2 * lambda(p,1) - 1);
    phip(p,2) = lambda(p,2) .* (2 * lambda(p,2) - 1);
    phip(p,3) = lambda(p,3) .* (2 * lambda(p,3) - 1);
    phip(p,4) = 4 * lambda(p,2) .* lambda(p,3);
    phip(p,5) = 4 * lambda(p,3) .* lambda(p,1);
end

% Assemble mass matrix M for P2 elements
sM = zeros(21*NT, nQuad);
for p = 1:nQuad
    index = 0;
    for i = 1:6
        for j = i:6
            Mij = weight(p) * phip(p,i) .* phip(p,j);
            Mij = Mij .* area(:, i);
            sM(index+1:index+NT,p) = Mij;
            index = index + NT;
        end
    end
end
sM = sum(sM, 2);

% Assemble the mass matrix
diagIdx = (ii == jj);
upperIdx = ~diagIdx;
M = sparse(ii(diagIdx), jj(diagIdx), sM(diagIdx), Ndof, Ndof);
MU = sparse(ii(upperIdx), jj(upperIdx), sM(upperIdx), Ndof, Ndof);
M = M + MU + MU';

% Energy calculation setup
F_energy = @(u) (1/4) * (u.^2 - 1).^2;
energy_grad = 0;
energy_potential = 0;

% Initial energy calculation
for p = 1:nQuad
    % 计算积分点在全局坐标中的位置
    pxy = lambda(p, 1) * node(elem(:, 1), :) ...
        + lambda(p, 2) * node(elem(:, 2), :) ...
        + lambda(p, 3) * node(elem(:, 3), :);

    % 初始化时直接用积分点位置计算初值
    u_p = 0.05 * sin(pxy(:, 1)) .* sin(pxy(:, 2));

    % 计算非线性势能项 F(u_p)
    fp = F_energy(u_p); % NT x1

    % 计算每个单元的梯度 grad_u
    grad_u_x = Dphip(:,1,1,p) .* u_all(elem2dof(:,1), 1) + ...
              Dphip(:,1,2,p) .* u_all(elem2dof(:,2), 1) + ...
              Dphip(:,1,3,p) .* u_all(elem2dof(:,3), 1) + ...
              Dphip(:,1,4,p) .* u_all(elem2dof(:,4), 1) + ...
              Dphip(:,1,5,p) .* u_all(elem2dof(:,5), 1) + ...
              Dphip(:,1,6,p) .* u_all(elem2dof(:,6), 1); % NT x1

    grad_u_y = Dphip(:,2,1,p) .* u_all(elem2dof(:,1), 1) + ...
              Dphip(:,2,2,p) .* u_all(elem2dof(:,2), 1) + ...
              Dphip(:,2,3,p) .* u_all(elem2dof(:,3), 1) + ...
              Dphip(:,2,4,p) .* u_all(elem2dof(:,4), 1) + ...
              Dphip(:,2,5,p) .* u_all(elem2dof(:,5), 1) + ...
              Dphip(:,2,6,p) .* u_all(elem2dof(:,6), 1); % NT x1

    % 计算 |grad_u|^2
    grad_u_sq = grad_u_x.^2 + grad_u_y.^2; % NT x1

    % 累加梯度能量项
    energy_grad = energy_grad + weight(p) * (epsilon^2 / 2) * grad_u_sq .* area(:, 1);

    % 累加势能项
    energy_potential = energy_potential + weight(p) * fp .* area(:, 1);
end

% 总能量
energy(1) = sum(energy_grad) + sum(energy_potential);

% 开始模拟
for n = 1:numSteps
    % 计算非线性项 F(u^n)
    f = @(u) u.^3 - u; % 定义非线性函数 f(u)
    F_local = zeros(NT, 6); % 初始化局部右端项

    % 循环计算每个积分点的贡献
    for p = 1:nQuad
        % 计算积分点在全局坐标中的位置
        pxy = lambda(p, 1) * node(elem(:, 1), :) ...
            + lambda(p, 2) * node(elem(:, 2), :) ...
            + lambda(p, 3) * node(elem(:, 3), :);

        % 获取当前时间步解 u 在每个自由度上的值
        u_current = u_all(:,n);
        u_vals = u_current(elem2dof(:,1:6)); % NT x6

        % Compute u_p using shape functions phip
        u_p = sum(phip(p,:) .* u_vals, 2); % NT x 1

        % 计算非线性项 f(u_p) 的值
        fp = f(u_p); % NT x1

        % 累加每个基函数在当前积分点的贡献
        for i = 1:6
            F_local(:, i) = F_local(:, i) + weight(p) * phip(p, i) .* fp;
        end
    end

    % 考虑每个单元的面积并累加到全局右端项
    F_local = F_local .* area(:, 1:6); % NT x6
    F = accumarray(elem2dof(:), F_local(:), [Ndof, 1]); % 累加得到全局右端项

    % 构造整体矩阵和右端项以一次性求解 u^{n+1} 和 w^{n+1}
    % 构造左端矩阵 A 和右端向量 b
    A = [M/dt, S; S, -M];
    b = [(M/dt) * u_all(:, n); -F / epsilon^2];

    % 求解线性系统 A * [u^{n+1}; w^{n+1}] = b
    sol = A \ b;

    % 更新 u 和 w
    u_all(:, n + 1) = sol(1:Ndof);
    w = sol(Ndof+1:end);

    % 计算当前时间步的能量
    energy_grad = 0;
    energy_potential = 0;
    for p = 1:nQuad
        % 获取当前时间步解 u 在每个自由度上的值
        u_current = u_all(:,n+1);
        u_vals = u_current(elem2dof);

        % Compute u_p using shape functions phip
        u_p = sum(phip(p,:) .* u_vals, 2); % NT x 1

        % 计算非线性势能项 F(u_p)
        fp = F_energy(u_p); % NT x1

        % 计算每个单元的梯度 grad_u
        grad_u_x = Dphip(:,1,1,p) .* u_all(elem2dof(:,1), n+1) + ...
                  Dphip(:,1,2,p) .* u_all(elem2dof(:,2), n+1) + ...
                  Dphip(:,1,3,p) .* u_all(elem2dof(:,3), n+1) + ...
                  Dphip(:,1,4,p) .* u_all(elem2dof(:,4), n+1) + ...
                  Dphip(:,1,5,p) .* u_all(elem2dof(:,5), n+1) + ...
                  Dphip(:,1,6,p) .* u_all(elem2dof(:,6), n+1); % NT x1

        grad_u_y = Dphip(:,2,1,p) .* u_all(elem2dof(:,1), n+1) + ...
                  Dphip(:,2,2,p) .* u_all(elem2dof(:,2), n+1) + ...
                  Dphip(:,2,3,p) .* u_all(elem2dof(:,3), n+1) + ...
                  Dphip(:,2,4,p) .* u_all(elem2dof(:,4), n+1) + ...
                  Dphip(:,2,5,p) .* u_all(elem2dof(:,5), n+1) + ...
                  Dphip(:,2,6,p) .* u_all(elem2dof(:,6), n+1); % NT x1

        % 计算 |grad_u|^2
        grad_u_sq = grad_u_x.^2 + grad_u_y.^2; % NT x1

        % 累加梯度能量项
        energy_grad = energy_grad + weight(p) * (epsilon^2 / 2) * grad_u_sq .* area(:, p);

        % 累加势能项
        energy_potential = energy_potential + weight(p) * fp .* area(:, p);
    end
    energy(n + 1) = sum(energy_grad) + sum(energy_potential);
end

% plot
figure;
for n = 1:1000:numSteps+1
    trisurf(elem, node(:, 1), node(:, 2), u_all(1:N, n));
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


