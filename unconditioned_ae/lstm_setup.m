function Weight=lstm_setup(layers)
%one layer considered
randnum=0.01;
M=layers(1);
N=layers(2);
% input gates
Weight.w_i=1/M*[normrnd(0,randnum,[M,N]);zeros(1,N)];
Weight.r_i=1/N*normrnd(0,randnum,[N,N]);
Weight.p_i=zeros(1,N);
% forget gates
Weight.w_f=1/M*[normrnd(0,randnum,[M,N]);ones(1,N)];
Weight.r_f=1/N*normrnd(0,randnum,[N,N]);
Weight.p_f=zeros(1,N);
% cells
Weight.w_z=1/M*[normrnd(0,randnum,[M,N]);zeros(1,N)];
Weight.r_z=1/N*normrnd(0,randnum,[N,N]);
% output gates
Weight.w_o=1/M*[normrnd(0,randnum,[M,N]);zeros(1,N)];
Weight.r_o=1/N*normrnd(0,randnum,[N,N]);
Weight.p_o=zeros(1,N);
% tanh layer
M=layers(2);
N=layers(3);
Weight.w_k=1/M*normrnd(0,randnum,[M,N]);
Weight.b_k=zeros(1,N);
