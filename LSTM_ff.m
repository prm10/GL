function [err,delta_y]=LSTM_ff(varargin)
% clc;clear;close all;
if nargin==0
    rand('seed',3);
    args.input=rand(10,3);
    args.label=[mean(args.input,2)<=0.5 mean(args.input,2)>0.5];
    args.numblocks=5;
    [T,M]=size(args.input);
    N=args.numblocks;
    outdims=size(args.label,2);

    %% weight initial
    % input gates
    w_i=rand(M+1,N);
    r_i=rand(N,N);
    p_i=rand(1,N);
    % forget gates
    w_f=rand(M+1,N);
    r_f=rand(N,N);
    p_f=rand(1,N);
    % cells
    w_z=rand(M+1,N);
    r_z=rand(N,N);
    % output gates
    w_o=rand(M+1,N);
    r_o=rand(N,N);
    p_o=rand(1,N);
    % output
    w_k=rand(N,outdims);
else
    args=varargin{1};
    delta=varargin{2};
    post=varargin{3};
    pos=varargin{4};
    [T,M]=size(args.input);
    N=args.numblocks;
    outdims=size(args.label,2);
    %% weight initial
    % input gates
    w_i=args.Weight.w_i;
    r_i=args.Weight.r_i;
    p_i=args.Weight.p_i;
    % forget gates
    w_f=args.Weight.w_f;
    r_f=args.Weight.r_f;
    p_f=args.Weight.p_f;
    % cells
    w_z=args.Weight.w_z;
    r_z=args.Weight.r_z;
    % output gates
    w_o=args.Weight.w_o;
    r_o=args.Weight.r_o;
    p_o=args.Weight.p_o;
    % output
    w_k=args.Weight.w_k;
end

label=args.label;
x=[args.input ones(T,1)];
%% value initial
in1=zeros(T,N);
in2=zeros(T,N);
f1=zeros(T,N);
f2=zeros(T,N);
z1=zeros(T,N);
z1=zeros(T,N);
c=zeros(T,N);
o1=zeros(T,N);
o2=zeros(T,N);
%% first time-step
t=1;
% input gates
in1(t,:)=x(t,:)*w_i;
in2(t,:)=sigmoid(in1(t,:));
% forget gates
f1(t,:)=x(t,:)*w_f;
f2(t,:)=sigmoid(f1(t,:));
% cells
z1(t,:)=x(t,:)*w_z;
z2(t,:)=tanh(z1(t,:));
c(t,:)=in2(t,:).*z2(t,:);
% output gates
o1(t,:)=x(t,:)*w_o+c(t,:).*p_o;
o2(t,:)=sigmoid(o1(t,:));
y(t,:)=o2(t,:).*tanh(c(t,:));
%% the rest time-step
for t=2:T
    % input gates
    in1(t,:)=x(t,:)*w_i+y(t-1,:)*r_i+c(t-1,:).*p_i;
    in2(t,:)=sigmoid(in1(t,:));
    % forget gates
    f1(t,:)=x(t,:)*w_f+y(t-1,:)*r_f+c(t-1,:).*p_f;
%     if t==T
%         f1(post,pos)=f1(post,pos)+delta;
%     end
    f2(t,:)=sigmoid(f1(t,:));
    % cells
    z1(t,:)=x(t,:)*w_z+y(t-1,:)*r_z;
    z2(t,:)=tanh(z1(t,:));
    c(t,:)=f2(t,:).*c(t-1,:)+in2(t,:).*z2(t,:);
    % output gates
    o1(t,:)=x(t,:)*w_o+y(t-1,:)*r_o+c(t,:).*p_o;
    o2(t,:)=sigmoid(o1(t,:));
    y(t,:)=o2(t,:).*tanh(c(t,:));
end
y(post,pos)=y(post,pos)+delta;
temp=y*w_k;
temp=exp(temp-max(temp,[],2)*ones(1,size(temp,2)));
data_out =temp./(sum(temp,2)*ones(1,size(temp,2)));
err=-sum(sum(label.* log(data_out)));
%% ���򴫲�
t=T;
delta_k=-(label-data_out);
delta_y(t,:)=delta_k(t,:)*w_k';
delta_o(t,:)=delta_y(t,:).*tanh(c(t,:)).*dsigmoid(o2(t,:));
delta_c(t,:)=delta_y(t,:).*o2(t,:).*dtanh(tanh(c(t,:)))+p_o.*delta_o(t,:);
delta_f(t,:)=delta_c(t,:).*c(t-1,:).*dsigmoid(f2(t,:));
delta_i(t,:)=delta_c(t,:).*z2(t,:).*dsigmoid(in2(t,:));
delta_z(t,:)=delta_c(t,:).*in2(t,:).*dtanh(z2(t,:));
for t=T-1:-1:1
    delta_y(t,:)=delta_k(t,:)*w_k'+delta_z(t+1,:)*r_z'+delta_i(t+1,:)*r_i'+delta_f(t+1,:)*r_f'+delta_o(t+1,:)*r_o';
    delta_o(t,:)=delta_y(t,:).*tanh(c(t,:)).*dsigmoid(o2(t,:));
    delta_c(t,:)=delta_y(t,:).*o2(t,:).*dtanh(tanh(c(t,:)))+p_o.*delta_o(t,:)+p_i.*delta_i(t+1,:)...
        +p_f.*delta_f(t+1,:)+delta_c(t+1,:).*f2(t+1,:);
    if t==1
        delta_f(t,:)=0;
    else
        delta_f(t,:)=delta_c(t,:).*c(t-1,:).*dsigmoid(f2(t,:));
    end
    delta_i(t,:)=delta_c(t,:).*z2(t,:).*dsigmoid(in2(t,:));
    delta_z(t,:)=delta_c(t,:).*in2(t,:).*dtanh(z2(t,:));
end

dw_k=y'*delta_k;

dw_o=x'*delta_o;
dw_f=x'*delta_f;
dw_i=x'*delta_i;
dw_z=x'*delta_z;

dr_o=y(1:end-1,:)'*delta_o(2:end,:);
dr_f=y(1:end-1,:)'*delta_f(2:end,:);
dr_i=y(1:end-1,:)'*delta_i(2:end,:);
dr_z=y(1:end-1,:)'*delta_z(2:end,:);

dp_i=sum(c(1:end-1,:).*delta_i(2:end,:));
dp_f=sum(c(1:end-1,:).*delta_f(2:end,:));
dp_o=sum(c.*delta_o);

    function y=sigmoid(x)
        y=1./(1+exp(-x));
    end
    function y=dsigmoid(z)
        y=z.*(1-z);
    end
    function y=dtanh(z)
        y=1-z.^2;
    end
end