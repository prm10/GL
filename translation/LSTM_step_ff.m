function [x,in2,f2,z2,c,o2,y]=LSTM_step_ff(input,Weight)
    %% weight initial
    % input gates
    w_i=Weight.w_i;
    r_i=Weight.r_i;
    p_i=Weight.p_i;
    % forget gates
    w_f=Weight.w_f;
    r_f=Weight.r_f;
    p_f=Weight.p_f;
    % cells
    w_z=Weight.w_z;
    r_z=Weight.r_z;
    % output gates
    w_o=Weight.w_o;
    r_o=Weight.r_o;
    p_o=Weight.p_o;
    %% data
    T=size(input,1);
    N=size(w_i,2);
    x=[input ones(T,1)];
    %% value initial
    in1=zeros(T,N);
    in2=zeros(T,N);
    f1=zeros(T,N);
    f2=zeros(T,N);
    z1=zeros(T,N);
    z2=zeros(T,N);
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
function y=sigmoid(x)
    y=1./(1+exp(-x));
