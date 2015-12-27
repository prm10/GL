clc;close all;clear;
%{
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=1;%高炉编号
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
i2=4;%:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});

i0=17;
hotWindPress=data1(:,i0);
coldWind=data1(:,8);

md=zeros(length(hotWindPress),1);
sd=zeros(length(hotWindPress),1);
for i1=1:length(hotWindPress)
    index=(max(1,i1-720):i1);
    tempData=hotWindPress(index);
    md(i1)=median(tempData);
    sd(i1)=std(tempData);
end
sHWP=(hotWindPress-md)./max(sd,0.0001);
dHWP=hotWindPress(2:end,:)-hotWindPress(1:end-1,:);
dHWP=[0;dHWP/std(dHWP)];

[indexChange]=FindStoveChange(hotWindPress);
figure;
subplot(311);
plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');
subplot(312);
plot(find(~indexChange),dHWP(~indexChange),'b.',find(indexChange),dHWP(indexChange),'r.');
subplot(313);
plot(find(~indexChange),sHWP(~indexChange),'b.',find(indexChange),sHWP(indexChange),'r.');
% figure;
% plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');

delay=50;
hotWindPressLabel=[false(delay,1);indexChange(1:end-delay,1)];
save('fscData.mat','hotWindPress','hotWindPressLabel','dHWP','sHWP','delay');
%}
%%
% load('fscData.mat');

% range=1:1000;
% figure;
% subplot(211);
% plot(hotWindPress(range,:));
% subplot(212);
% plot(hotWindPressLabel(range,:));

%
load('fscDataHWP.mat');
global train_data train_label test_data test_label;
train_data=cell(0);
train_label=cell(0);
train_len=size(dataTrain,1);
test_len=size(dataTest,1);
rng(3);
lenInput=1000;
num=50;
index=floor(rand(num,1)*(train_len-lenInput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
%     range1=1:size(dHWP,1);
    train_data=[train_data;[dataTrain(range1,:)]];
    a=labelTrain(range1,:);
    a(1+delay:lenInput,1)=a(1:lenInput-delay,1);
    a(1:delay,1)=false;
    train_label=[train_label;[a,~a]];
end

test_data=cell(0);
test_label=cell(0);
num=10;
index=floor(rand(num,1)*(test_len-lenInput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
%     range1=1:size(dHWP,1);
    test_data=[test_data;[dataTest(range1,:)]];
    a=labelTest(range1,:);
    a(1+delay:lenInput,1)=a(1:lenInput-delay,1);
    a(1:delay,1)=false;
    test_label=[test_label;[a,~a]];
end

% 参数设置
args_name='args_fsc.mat';
choice=1;
switch choice
    case 1%初始化
        args.maxecho=10;
        args.circletimes=10;
        args.momentum=0.9;
        args.weightDecay=1e-5;
        args.learningrate=1e-1;
        args.batchsize=3;
        args.layer=[1,10,2];
        args.Er=[];
        args.outputLayer='softmax';
        args=fsc_initial(args);
        [args]=fsc_train(args);
        save(args_name,'args');
    case 2%继续运算
        load(args_name);
        args.maxecho=20;
        args.circletimes=100;
%         args.momentum=0.5;
%         args.learningrate=5e-2;
        args.batchsize=3;
        [args]=fsc_train(args);
        save(args_name,'args');
    case 3%梯度检验
        load(args_name);
        args.maxecho=1;
        args.circletimes=1;
        args.momentum=0;
        args.learningrate=0;
        [args]=fsc_train(args);%计算下当前的梯度
        vname='args.Weight{1, 1}.w_i';
        vname=vname(6:end);
        s1=strcat('dcal=args.Mom.',vname,'(1,1);');
        eval(s1);
    %     dcal=args.Mom.WeightPredict{1, 1}.w_i(1,1);
        error_delta=1e-10/dcal;%1e-5;%
        s2=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)+error_delta;');
        eval(s2);
    %     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)+error_delta;
        [~,error1]=fsc_ff(train_data,train_label,args);
        s3=strcat('args.',vname,'(1,1)=args.',vname,'(1,1)-2*error_delta;');
        eval(s3);
    %     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)-2*error_delta;
        [~,error2]=fsc_ff(train_data,train_label,args);
        dreal=(error1-error2)/error_delta/2;
        accuracy=abs((dcal-dreal)/dreal)*100;
end

figure;
plot((1:length(args.Er))*100,args.Er);
title('误差下降曲线');
xlabel('迭代次数');
ylabel('误差');

i1=1;
[predict,error2]=fsc_ff(test_data(i1),test_label(i1),args);
data=test_data{i1}(1:end-delay,1);
range1=test_label{i1}(delay+1:end,1);
predict_label=predict>0.5;
range2=predict_label(delay+1:end,1);
figure;
subplot(311);
plot(find(~range1),data(~range1),'b.',find(range1),data(range1),'r.');
subplot(312);
plot(find(~range2),data(~range2),'b.',find(range2),data(range2),'r.');
subplot(313);
plot(predict(delay+1:end,1));
%}

%%
%{
load('fscData.mat');
load 'args_fsc_1221_best.mat';
[predict,~]=fsc_ff({dHWP},{[hotWindPressLabel,~hotWindPressLabel]},args);
data=hotWindPress(1:end-delay,1);
range1=hotWindPressLabel(delay+1:end,1); 
predict_label=predict>0.7;
range2=predict_label(delay+1:end,1);
figure;
subplot(311);
plot(find(~range1),data(~range1),'b.',find(range1),data(range1),'r.');
subplot(312);
plot(find(~range2),data(~range2),'b.',find(range2),data(range2),'r.');
subplot(313);
plot(predict(delay+1:end,1));
%}