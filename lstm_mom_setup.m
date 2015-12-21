function Weight=lstm_mom_setup(layers)
%% Momentum initial
n=length(layers)-1;%lstm+tanhµÄ²ãÊý
Weight=cell(n,1);
for i1=1:n-1
    M=layers(i1);
    N=layers(i1+1);
    % input gates
    Weight{i1}.w_i=zeros(M+1,N);
    Weight{i1}.r_i=zeros(N,N);
    Weight{i1}.p_i=zeros(1,N);
    % forget gates
    Weight{i1}.w_f=zeros(M+1,N);
    Weight{i1}.r_f=zeros(N,N);
    Weight{i1}.p_f=zeros(1,N);
    % cells
    Weight{i1}.w_z=zeros(M+1,N);
    Weight{i1}.r_z=zeros(N,N);
    % output gates
    Weight{i1}.w_o=zeros(M+1,N);
    Weight{i1}.r_o=zeros(N,N);
    Weight{i1}.p_o=zeros(1,N);
end
% translation
M=layers(n);
N=layers(n+1);
Weight{n}.w_k=zeros(M,N);
Weight{n}.b_k=zeros(1,N);
