function [x2,in2,f2,z2,c2,o2,y2,output]=LSTM_step_ff_de(T,x1,c0,C,w_i,r_i,p_i,w_f,r_f,p_f,w_z,r_z,w_o,r_o,p_o,w_k,b_k)
%{
假设为一层的LSTM
输入参数：
    x1：t=1的输入，一行
    c0：初始状态
    W：权值，包括x和b
输出参数：
    x2,in2,f2,z2,c2,o2,y2
    output
%}

N0=size(C,2);
N1=size(x1,2);
N2=size(w_i,2);
N3=size(b_k,2);
%% data
x2=[zeros(T,N0+N1) ones(T,1)];
in1=zeros(T,N2);
in2=zeros(T,N2);
f1=zeros(T,N2);
f2=zeros(T,N2);
z1=zeros(T,N2);
z2=zeros(T,N2);
c2=zeros(T,N2);
o1=zeros(T,N2);
o2=zeros(T,N2);
output=zeros(T,N3);
%% first time-step
t=1;
x2(t,1:end-1)=[C,x1];
% input gates
in1(t,:)=x2(t,:)*w_i+c0.*p_i;
in2(t,:)=sigmoid(in1(t,:));
% forget gates
f1(t,:)=x2(t,:)*w_f+c0.*p_f;
f2(t,:)=sigmoid(f1(t,:));
% cells
z1(t,:)=x2(t,:)*w_z;
z2(t,:)=tanh(z1(t,:));
c2(t,:)=f2(t,:).*c0+in2(t,:).*z2(t,:);
% output gates
o1(t,:)=x2(t,:)*w_o+c2(t,:).*p_o;
o2(t,:)=sigmoid(o1(t,:));
y2(t,:)=o2(t,:).*tanh(c2(t,:));
output(t,:)=tanh(y2(t,:)*w_k+b_k);
%% the rest time-step
for t=2:T
    x2(t,1:end-1)=[C,output(t-1,:)];
    % input gates
    in1(t,:)=x2(t,:)*w_i+y2(t-1,:)*r_i+c2(t-1,:).*p_i;
    in2(t,:)=sigmoid(in1(t,:));
    % forget gates
    f1(t,:)=x2(t,:)*w_f+y2(t-1,:)*r_f+c2(t-1,:).*p_f;
    f2(t,:)=sigmoid(f1(t,:));
    % cells
    z1(t,:)=x2(t,:)*w_z+y2(t-1,:)*r_z;
    z2(t,:)=tanh(z1(t,:));
    c2(t,:)=f2(t,:).*c2(t-1,:)+in2(t,:).*z2(t,:);
    % output gates
    o1(t,:)=x2(t,:)*w_o+y2(t-1,:)*r_o+c2(t,:).*p_o;
    o2(t,:)=sigmoid(o1(t,:));
    y2(t,:)=o2(t,:).*tanh(c2(t,:));
    output(t,:)=tanh(y2(t,:)*w_k+b_k);
    % if mod(t,ceil(T/10))==0
    %     disp(strcat(num2str(t/T*10,2),'%'));
    % end
end

function y=sigmoid(x)
    y=1./(1+exp(-x));
