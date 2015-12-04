clc;close all;clear all;
rng(3);
T=500;
t=[1:T]';
x1=(sin(t/pi)+sin(t*0.8/pi))/3;%normrnd(0,0.1,[T,1])+
% plot(t,x1,'.');
% hist(x1,100)
global train_data train_label test_data test_label;
train_data=cell(0);
train_label=cell(0);
lenInput=100;
lenOutput=10;
num=10;
index=floor(rand(num,1)*(size(x1,1)-lenInput-lenOutput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    range2=index(i1)+lenInput+1:index(i1)+lenInput+lenOutput;
    train_data=[train_data;x1(range1,:)];
    train_label=[train_label;x1(range2,:)];
end
test_data=train_data;
test_label=train_label;
% 参数设置
% load('args.mat');
if(exist('args','var'))%梯度检查
    args.maxecho=1;
    args.circletimes=1;
    args.momentum=0;
    args.learningrate=0;
    [args]=LSTM_train(args);%计算下当前的梯度
    vname='args.WeightTranP{1, 1}.w_k';
    vname=vname(6:end);
    s1=strcat('dcal=args.Mom.',vname,'(1,1);');
    eval(s1);
%     dcal=args.Mom.WeightPredict{1, 1}.w_i(1,1);
    error_delta=1e-11/dcal;%1e-5;%
    s2=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)+error_delta;');
    eval(s2);
%     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)+error_delta;
    [~,~,errorR1,errorP1]=LSTM_ff(train_data,train_label,args);
    s3=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)-2*error_delta;');
    eval(s3);
%     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)-2*error_delta;
    [~,~,errorR2,errorP2]=LSTM_ff(train_data,train_label,args);
    dreal=(errorR1+errorP1-errorR2-errorP2)/error_delta/2;
    accuracy=abs((dcal-dreal)/dreal)*100;
else %初始化
    args.maxecho=400;
    args.circletimes=100;
    args.momentum=0.9;
    args.learningrate=1e-1;
    args.batchsize=10;
    dimC=10;
    dimInput=1;
    dimOutput=1;
    layer=[50];
    args.encoderLayer=[1,layer,dimC];
    args.decoderLayer=[dimC+dimInput,layer,dimInput];
    args.predictLayer=[dimC+dimOutput,layer,dimOutput];
    args=LSTM_initial(args);
    [args]=LSTM_train(args);
    save('args.mat','args');
%     test_data=train_data;
%     test_label=train_label;
%     [reconstruct,predict,errorR,errorP]=LSTM_ff(test_data,test_label,args);
end

i1=5;
[reconstruct,predict,errorR,errorP]=LSTM_ff(test_data(i1),{zeros(100,1)},args);
figure;
plot(1:size(test_data{i1},1),test_data{i1},'--o'...
    ,size(test_data{i1},1)+1:size(test_data{i1},1)+size(test_label{i1},1),test_label{i1},'--o'...
    ,1:size(reconstruct,1),reconstruct(end:-1:1,:),'*'...
    ,size(reconstruct,1)+1:size(reconstruct,1)+size(predict,1),predict,'*');
