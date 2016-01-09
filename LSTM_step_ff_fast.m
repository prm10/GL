% function [x2,in2,f2,z2,c2,o2,y2,output]=LSTM_step_ff_fast(input_x11,input_c0,input_xC,Weight,T,outputLayer)
function [x2,in2,f2,z2,c2,o2,y2,output]=LSTM_step_ff_fast(x1,c0,outputLayer,...
    w_i,r_i,p_i,...
    w_f,r_f,p_f,...
    w_z,r_z,...
    w_o,r_o,p_o,...
    w_k,b_k)
%{
假设为一层的LSTM
输入参数：
    x1：输入，列是样本维度，行是时间
    c0：初始状态
    W：权值，包括x和b
    Wk：输出层的权值
    outputLayer：输出层的类型：softmax和tanh两种可选
输出参数：
    x2,in2,f2,z2,c2,o2,y2,output
%}
%% data
T=size(x1,1);
N=size(w_i,2);
x2=[x1 ones(T,1)];
%% value initial
in1=zeros(T,N);
in2=zeros(T,N);
f1=zeros(T,N);
f2=zeros(T,N);
z1=zeros(T,N);
z2=zeros(T,N);
c2=zeros(T,N);
o1=zeros(T,N);
o2=zeros(T,N);
%% first time-step
t=1;
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

%% the rest time-step
for t=2:T
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
end

%最后一层
z_k2=y2*w_k+ones(T,1)*b_k;
numClass=size(w_k,2);
switch outputLayer
    case{'softmax'}
        temp=exp(z_k2-max(z_k2,[],2)*ones(1,numClass));
        output=temp./(sum(temp,2)*ones(1,numClass));
    otherwise
        output=tanh(z_k2);
end
function y=sigmoid(x)
    y=1./(1+exp(-x));