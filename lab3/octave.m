pkg load interval
addpath(genpath('./m'))
addpath(genpath('./data'))

format long g

data1 = csvread("Ch1_800nm_0.03.csv")
data2 = csvread("Ch2_800nm_0.03.csv")
# remove first rows
data1(1,:) = []
data2(1,:) = []
#leave only mV values
data1_mv = data1(:,1)
data2_mv = data2(:,1)
# get N values
data1_n = transpose(1:length(data1_mv))
data2_n = transpose(1:length(data2_mv))

# get Epsilon
data1_eps = 1e-4
data2_eps = 1e-4

%setup a problem
data1_X = [ data1_n.^0 data1_n ];
data1_inf_b = data1_mv - data1_eps
data1_sup_b = data1_mv + data1_eps
[data1_tau, data1_w] = L_1_minimization(data1_X, data1_inf_b, data1_sup_b);

data2_X = [ data2_n.^0 data2_n ];
data2_inf_b = data2_mv - data2_eps
data2_sup_b = data2_mv + data2_eps
[data2_tau, data2_w] = L_1_minimization(data2_X, data2_inf_b, data2_sup_b);

fileID = fopen('data/Ch1.txt','w');
fprintf(fileID,'%g %g\n', data1_tau(1), data1_tau(2));
for c = 1 : length(data1_w)
  fprintf(fileID, "%g\n", data1_w(c));
end
fclose(fileID);

fileID = fopen('data/Ch2.txt','w');
fprintf(fileID,'%g %g\n', data2_tau(1), data2_tau(2));
for c = 1 : length(data2_w)
  fprintf(fileID, "%g\n", data2_w(c));
end
fclose(fileID);

