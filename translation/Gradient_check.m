clc;close all;clear all;
rng(3);
T=2000;
t=[1:T]';
x1=normrnd(0,0.1,[T,1])+(sin(t/pi)+sin(t*0.8/pi))/2;
% plot(t,x1);
% hist(x1,100)
global train_data train_label test_data test_label;
train_data=cell(0);
train_label=cell(0);
lenInput=10;
lenOutput=1;
num=1;
index=floor(rand(num,1)*(size(x1,1)-lenInput-lenOutput));
for i1=1:num
    range1=index(i1):index(i1)+lenInput;
    range2=index(i1)+lenInput+1:index(i1)+lenInput+lenOutput;
    train_data=[train_data;x1(range1,:)];
    train_label=[train_label;x1(range2,:)];
end
test_data=train_data(1);
test_label=train_label(1);
% 参数设置
load('args.mat');
if(exist('args','var'))%梯度检查
    args.circletimes=1;
    args.momentum=0;
    args.learningrate=0;
    [args]=LSTM_train(args);
    [~,~,errorR0,errorP0]=LSTM_ff(train_data,train_label,args);
    dcal=args.Mom.WeightPredict{1, 2}.w_k;
    delta=1e-6;
    args.WeightPredict{1, 2}.w_k=args.WeightPredict{1, 2}.w_k+delta;
    [~,~,errorR1,errorP1]=LSTM_ff(train_data,train_label,args);
    args.WeightPredict{1, 2}.w_k=args.WeightPredict{1, 2}.w_k-2*delta;
    [~,~,errorR2,errorP2]=LSTM_ff(train_data,train_label,args);
    dreal=(errorP1-errorP2)/delta/2;
else %初始化
    args.maxecho=1;
    args.circletimes=100;
    args.momentum=0.9;
    args.learningrate=1e-2;
    dimC=1;
    dimInput=1;
    dimOutput=1;
    args.encoderLayer=[1,1,dimC];
    args.decoderLayer=[dimC+dimInput,1,dimInput];
    args.predictLayer=[dimC+dimOutput,1,dimOutput];
    args=LSTM_initial(args);
    [args]=LSTM_train(args);
    save('args.mat','args');
    [reconstruct,predict,errorR,errorP]=LSTM_ff(train_data,train_label,args);
    plot(1:size(test_data{1},1),test_data{1},'--o'...
        ,size(test_data{1},1)+1:size(test_data{1},1)+size(test_label{1},1),test_label{1},'--o'...
        ,1:size(reconstruct,1),reconstruct(end:-1:1,:),'*'...
        ,size(reconstruct,1)+1:size(reconstruct,1)+size(predict,1),predict,'*');
end

