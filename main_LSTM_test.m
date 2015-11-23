clc;close all;clear all;
rng(3);
%% 输入：随机数及其带时延的操作；
% 输出：1：第1维输入的前3项x(t-1)的和；2：第2维减去第1维；
delay=4;
len=200;
for i1=1:len
    input{i1}=rand(100,3)-0.5;
%     l1=[ones(delay,1);mean(input{i1}(1:end-delay,:),2)<=0];
%     l2=[zeros(delay,1);mean(input{i1}(1:end-delay,:),2)>0];
%     label{i1}=[l1 l2];

    l1=[zeros(delay,1);input{i1}(1:end-delay,1)];
    l2=[zeros(2*delay,1);mean(input{i1}(1:end-2*delay,:),2)];
    l3=[-input{i1}(1:2*delay,1);input{i1}(1:end-2*delay,3)-input{i1}(2*delay+1:end,1)];
    label{i1}=[l3];
end
train_input=input(1:ceil(len/2));
train_label=label(1:ceil(len/2));

test_input=input(ceil(len/2):end);
test_label=label(ceil(len/2):end);
clear input label;
args.layer=[size(train_input{1},2) 20 size(train_label{1},2)];
args.maxecho=100;
args.momentum=0.9;
args.learningrate=1e-1;
args.outputtype='tanh';
args=LSTM_initial(args);
[args]=LSTM_train(args,train_input,train_label);
[dout,error]=LSTM_ff(test_input,test_label,args);

out=test_label{1,1};
predict=dout{1,1};
plot(1:size(out,1),out(:,1),1:size(predict,1),predict(:,1));


