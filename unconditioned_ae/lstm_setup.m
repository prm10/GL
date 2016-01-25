function Weight=lstm_setup(layers)
n=length(layers)-1;%lstm+tanh�Ĳ���
Weight=cell(n,1);
for i1=1:n-1
    M=layers(i1);
    N=layers(i1+1);
    % input gates
    Weight{i1}.w_i=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    Weight{i1}.r_i=1/N*normrnd(0,0.1,[N,N]);
    Weight{i1}.p_i=zeros(1,N);
    % forget gates
    Weight{i1}.w_f=1/M*[normrnd(0,0.1,[M,N]);ones(1,N)];
    Weight{i1}.r_f=1/N*normrnd(0,0.1,[N,N]);
    Weight{i1}.p_f=zeros(1,N);
    % cells
    Weight{i1}.w_z=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    Weight{i1}.r_z=1/N*normrnd(0,0.1,[N,N]);
    % output gates
    Weight{i1}.w_o=1/M*[normrnd(0,0.1,[M,N]);zeros(1,N)];
    Weight{i1}.r_o=1/N*normrnd(0,0.1,[N,N]);
    Weight{i1}.p_o=zeros(1,N);
end
M=layers(n);
N=layers(n+1);
Weight{n}.w_k=1/M*normrnd(0,0.1,[M,N]);
Weight{n}.b_k=zeros(1,N);
