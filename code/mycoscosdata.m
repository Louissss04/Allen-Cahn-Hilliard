function pde = mycoscosdata(dt)

    pde = struct('f', @f, 'exactu', @exactu, 'Du', @Du);
    
    function rhs = f(p)
        x = p(:,1); 
        y = p(:,2);
        rhs = (1 + (8*dt/3)*pi^4) .* cos(pi*x) .* cos(pi*y);
    end
    
    function u = exactu(p)
        x = p(:,1); 
        y = p(:,2);
        u = cos(pi*x) .* cos(pi*y);
    end

%     function rhs = f(p)
%         x = p(:,1); 
%         y = p(:,2);
%         rhs = 1;
%     end
%     
%     function u = exactu(p)
%         x = p(:,1); 
%         y = p(:,2);
%         u = 1;
%     end
    
    function uprime = Du(p)
        x = p(:,1); 
        y = p(:,2);
        uprime(:,1) = -pi * sin(pi*x) .* cos(pi*y); 
        uprime(:,2) = -pi * cos(pi*x) .* sin(pi*y); 
    end
%     function uprime = Du(p)
%         x = p(:,1); 
%         y = p(:,2);
%         uprime(:,1) = 0; 
%         uprime(:,2) = 0; 
%     end
end
