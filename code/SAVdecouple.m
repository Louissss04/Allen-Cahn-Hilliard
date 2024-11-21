function [u, w, L2_error, H1_error] = SAVdecouple(node, elem, pde, dt)

    %% 1. 定义 PDE 的真解和计算右端项 f
    exact_u = pde.exactu;
    exact_Du = pde.Du;
    f = pde.f;
    
    %% 2. 初始化质量矩阵 M 和 刚度矩阵 S
    N = size(node, 1); 
    NT = size(elem, 1);
    Ndof = N;
    
    M = sparse(Ndof, Ndof);
    S = sparse(Ndof, Ndof);
    
    %% 3. 组装刚度矩阵 S
    [Dphi, area] = gradbasis(node, elem);
    
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

    %% 4. 组装质量矩阵 M
%     for i = 1:3
%         for j = i:3
%             if i == j
%                 Mij = (1/6) * area; % 对角部分
%             else
%                 Mij = (1/12) * area; % 非对角部分
%             end
%             M = M + sparse([elem(:, i); elem(:, j)], [elem(:, j); elem(:, i)], [Mij; Mij], Ndof, Ndof);
%         end
%     end
    for i = 1:3
        for j = i:3
            if i == j
                Mij = (1/6) * area; % 对角部分
                
                M = M + sparse(elem(:, i), elem(:, j), Mij, Ndof, Ndof);
            else
                Mij = (1/12) * area; % 非对角部分
              
                M = M + sparse([elem(:, i); elem(:, j)], [elem(:, j); elem(:, i)], [Mij; Mij], Ndof, Ndof);
            end
        end
    end

   
    %% 5. 组装块矩阵 A 和 块载荷向量 b
    % A = [M, -2*dt/3 * S;
    %      S,  M]
    A = [M, 2*dt/3 * S;
         -S, M];
    
    % 6. 组装载荷向量 b = [f; 0]
    [lambda, weight] = quadpts(3); 
    nQuad = size(lambda, 1);
    
    % 初始化局部载荷向量 bt
    bt = zeros(NT, 3);
    
    for p = 1:nQuad
        % 计算 quadrature 点的坐标
        pxy = lambda(p,1)*node(elem(:,1),:) + ...
              lambda(p,2)*node(elem(:,2),:) + ...
              lambda(p,3)*node(elem(:,3),:);
        
        % 计算 f 在 quadrature 点的值
        fp = f(pxy); % NT x1
        
        % 组装每个基函数的贡献
        for i = 1:3
            bt(:,i) = bt(:,i) + weight(p)*lambda(p,i).*fp;
        end
    end
    
    % 乘以单元面积
    bt = bt .* repmat(area, 1, 3); % NT x3
    
    % 使用 accumarray 组装全局载荷向量 b
    b = accumarray(elem(:), bt(:), [Ndof, 1]); % N x1
    
    % 右端项 b = [f; 0]
    b = [b; zeros(Ndof, 1)];
    
    %% 7. 求解线性系统 A * sol = b
    sol = A \ b;
    
    %% 8. 提取解向量 u 和 w
    u = sol(1:Ndof);
    w = sol(Ndof+1:end);
    
    %% 9. 计算误差
    L2_error = getL2error(node, elem, exact_u, u);
    H1_error = getH1error(node, elem, exact_Du, u); 
end
