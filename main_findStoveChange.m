clc;close all;clear;
%{
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=1;
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
i2=4;%:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});

i0=17;
hotWindPress=data1(:,i0);
coldWind=data1(:,8);

% md=zeros(length(hotWindPress),1);
% sd=zeros(length(hotWindPress),1);
% for i1=1:length(hotWindPress)
%     index=(max(1,i1-720):i1);
%     tempData=hotWindPress(index);
%     md(i1)=median(tempData);
%     sd(i1)=std(tempData);
% end
% flag=(hotWindPress-md)./sd;
flag=hotWindPress(2:end,:)-hotWindPress(1:end-1,:);
flag=[0;flag/std(flag)];

[indexChange]=FindStoveChange(hotWindPress);
figure;
subplot(311);
plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');
subplot(312);
plot(find(~indexChange),flag(~indexChange),'b.',find(indexChange),flag(indexChange),'r.');
subplot(313);
plot(find(~indexChange),coldWind(~indexChange),'b.',find(indexChange),coldWind(indexChange),'r.');
% figure;
% plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');


delay=50;
hotWindPress=flag;
hotWindPressLabel=[false(delay,1);indexChange(1:end-delay,1)];
save('fscData.mat','hotWindPress','hotWindPressLabel','delay');
%}
%%
load('fscData.mat');
range=1:1000;
% figure;
% subplot(211);
% plot(hotWindPress(range,:));
% subplot(212);
% plot(hotWindPressLabel(range,:));

global train_data train_label test_data test_label;
train_data=cell(0);
train_label=cell(0);
train_len=2e4;
test_len=size(hotWindPress,1)-train_len;
rng(3);
lenInput=1000;
num=100;
index=floor(rand(num,1)*(train_len-lenInput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    train_data=[train_data;hotWindPress(range1,:)];
    train_label=[train_label;[hotWindPressLabel(range1,:),~hotWindPressLabel(range1,:)]];
end

test_data=cell(0);
test_label=cell(0);
num=10;
index=floor(rand(num,1)*(test_len-lenInput))+train_len;
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    test_data=[test_data;hotWindPress(range1,:)];
    test_label=[test_label;[hotWindPressLabel(range1,:),~hotWindPressLabel(range1,:)]];
end

% 参数设置
args_name='args_fsc.mat';
choice=2;
switch choice
    case 1%初始化
        args.maxecho=10;
        args.circletimes=10;
        args.momentum=0.99;
        args.learningrate=1e-1;
        args.batchsize=4;
        args.layer=[1,50,2];
        args.outputLayer='softmax';
        args=fsc_initial(args);
        [args]=fsc_train(args);
        save(args_name,'args');
    case 2%继续运算
        load(args_name);
        args.maxecho=100;
        args.circletimes=100;
        args.momentum=0.9;
        args.learningrate=1e-1;
        args.batchsize=4;
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



