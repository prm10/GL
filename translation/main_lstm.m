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
lenInput=200;
lenOutput=10;
num=10;
index=floor(rand(num,1)*(size(x1,1)-lenInput-lenOutput));
for i1=1:num
    range1=index(i1):index(i1)+lenInput;
    range2=index(i1)+lenInput+1:index(i1)+lenInput+lenOutput;
    train_data=[train_data;x1(range1,:)];
    train_label=[train_label;x1(range2,:)];
end
test_data=train_data(1);
test_label=train_label(1);
% ≤Œ ˝…Ë÷√
% load('args.mat');
if(exist('args','var'))
    
else
    args.maxecho=10;
    args.circletimes=100;
    args.momentum=0.9;
    args.learningrate=1e-5;
    % args.predictLength=1000;
    dimC=10;
    dimInput=1;
    dimOutput=1;
    args.encoderLayer=[1,100,dimC];
    args.decoderLayer=[dimC+dimOutput,100,dimInput];
    args.predictLayer=[dimC+dimOutput,100,dimOutput];
    args=LSTM_initial(args);
end
[args]=LSTM_train(args);
save('args.mat','args');
[reconstruct,predict,errorR,errorP]=LSTM_ff(test_data,test_label,args);
plot(1:size(test_data{1},1),test_data{1},'--',size(test_data{1},1)+1:size(test_data{1},1)+size(test_label{1},1),test_label{1},'--',1:size(reconstruct,1),reconstruct(end:-1:1,:),size(reconstruct,1)+1:size(reconstruct,1)+size(predict,1),predict);
