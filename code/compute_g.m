function g = compute_g(u, epsilon)
    % compute_g 计算源项 g 的匿名函数句柄
    %
    % 输入:
    %   u       - 输入的匿名函数句柄，形式为 @(x, y, t) ...
    %   epsilon - 参数 ε
    %
    % 输出:
    %   g       - 输出的匿名函数句柄，形式为 @(x, y, t) ...

    % 确保 Symbolic Math Toolbox 可用
    if ~license('test', 'symbolic_toolbox')
        error('Symbolic Math Toolbox is required.');
    end

    % 定义符号变量
    syms x y t

    % 将匿名函数句柄转换为符号表达式
    % 注意：u 必须能够接受符号变量作为输入
    try
        u_sym = u(x, y, t);
    catch
        error('输入的匿名函数 u 必须能够接受符号变量 x, y, t 作为输入。');
    end

    % 计算时间导数 u_t
    u_t = diff(u_sym, t);

    % 计算空间拉普拉斯算子 Δu = u_xx + u_yy
    u_xx = diff(u_sym, x, 2);
    u_yy = diff(u_sym, y, 2);
    laplace_u = u_xx + u_yy;

    % 定义非线性项 f(u) = u^3 - u
    f_u = u_sym^3 - u_sym;

    % 计算 term = -Δu + (1 / epsilon^2) * f(u)
    term = -laplace_u + (1 / epsilon^2) * f_u;

    % 计算 Δ(term) = term_xx + term_yy
    term_xx = diff(term, x, 2);
    term_yy = diff(term, y, 2);
    laplace_term = term_xx + term_yy;

    % 计算源项 g = u_t - Δ(term)
    g_sym = u_t - laplace_term;

    % 简化表达式
    g_sym = simplify(g_sym);

    % 将符号表达式转换为匿名函数句柄
    g = matlabFunction(g_sym, 'Vars', {x, y, t});
end
