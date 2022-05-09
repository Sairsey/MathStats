function [tau, w] = L_1_minimization(A, inf_b, sup_b)
  
  [m, n] = size(A);
  
  rad_b = (sup_b - inf_b) / 2;
  mid_b = (inf_b + sup_b) / 2;
  
  c = [zeros(n, 1); ones(m, 1)];
  
  M_1 = [A, -diag(rad_b)];
  M_2 = [A, diag(rad_b)];
  M = [M_1; M_2];
  % figure; spy(M_1)
  
  v_1 = mid_b;
  v_2 = mid_b;
  v = [v_1; v_2];
  
  c_type_1 = char(double("U") * ones(m, 1));
  c_type_2 = char(double("L") * ones(m, 1));
  c_type = [c_type_1; c_type_2];
  
  var_type = char(double("C") * ones(n + m, 1));
 
% non-negative  
 % l_b = [zeros(n, 1); ones(m, 1)];
  l_b = [-Inf(n, 1); ones(m, 1)];
  u_b = Inf(n + m, 1);
  
  y = glpk(c, M, v, l_b, u_b, c_type, var_type);
  
  tau = y(1:n);
  w = y((n + 1):(n + m));
  
endfunction
