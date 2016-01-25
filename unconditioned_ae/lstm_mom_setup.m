function Weight=lstm_mom_setup(layers)
%% Momentum initial
M=layers(1);
N=layers(2);
% input gates
Weight.w_i=zeros(M+1,N);
Weight.r_i=zeros(N,N);
Weight.p_i=zeros(1,N);
% forget gates
Weight.w_f=zeros(M+1,N);
Weight.r_f=zeros(N,N);
Weight.p_f=zeros(1,N);
% cells
Weight.w_z=zeros(M+1,N);
Weight.r_z=zeros(N,N);
% output gates
Weight.w_o=zeros(M+1,N);
Weight.r_o=zeros(N,N);
Weight.p_o=zeros(1,N);
% tanh layer
M=layers(2);
N=layers(3);
Weight.w_k=zeros(M,N);
Weight.b_k=zeros(1,N);
