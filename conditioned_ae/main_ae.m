clc;close all;clear;
%
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;
load(strcat('..\data\',num2str(No(i1)),'\data_labeled.mat'));
load(strcat('..\data\',num2str(No(i1)),'\sv.mat'));
i2=6;
data1=input0{i2}(:,commenDim{GL(i1)});
sv1=sv{i2};

i3=17;
hotWindPress=data1(~sv1,i3);
Mh=M(i3);
Sh=S(i3);
T=size(hotWindPress,1);
hotWindPress=(hotWindPress-ones(T,1)*Mh)./(ones(T,1)*Sh);
% hotWindPress=smooth(hotWindPress);
%{
figure;
subplot(211);
plot(data1(:,i3));
subplot(212);
plot(hotWindPress);
%}

%%
%
global train_data train_label test_data test_label;
train_data=cell(0);
train_label=cell(0);

train_len=ceil(size(hotWindPress,1)/2);
test_len=size(hotWindPress,1)-train_len;
dataTrain=hotWindPress(1:train_len,:);
dataTest=hotWindPress(1+train_len:test_len+train_len,:);
rng(11);
lenInput=6*20;
L=6*10;
num=1;
index=floor(rand(num,1)*(train_len-lenInput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    train_data=[train_data;[dataTrain(range1,:)]];
    train_label=[train_label;[dataTrain(range1(end:-1:end-L+1),:)]];
end

test_data=cell(0);
test_label=cell(0);
num=10;
index=floor(rand(num,1)*(test_len-lenInput));
for i1=1:num
    range1=index(i1)+1:index(i1)+lenInput;
    test_data=[test_data;[dataTest(range1,:)]];
    test_label=[test_label;[dataTest(range1(end:-1:end-L+1),:)]];
end

args_name='args_ae.mat';
choice=3;
switch choice
    case 1
        args.maxecho=1;
        args.circletimes=100;
        args.momentum=0.9;
        args.weightDecay=0;
        args.learningrate=1e-1;
        args.batchsize=1;
        args.layerEncoder=[1,200,3];
        args.layerStatic=[3 10];
        args.layerDecoder=[3+1,200,1];
        args.Er=[];
        args.outputLayer='tanh';
        args=ae_initial(args);
        [args]=ae_train(args);
        save(args_name,'args');
    case 2
        load(args_name);
        args.maxecho=10;
        args.circletimes=100;
%         args.momentum=0.9;
        args.learningrate=2e-2;
%         args.batchsize=3;
        [args]=ae_train(args);
        save(args_name,'args');
    case 3
        load(args_name);
        args.maxecho=1;
        args.circletimes=1;
        args.momentum=0;
        args.learningrate=0;
        args.weightDecay=0;
        
        [args]=ae_train(args);
        vname='args.WeightStatic.w_k2';
        loc='(end,end)';
        vname=vname(6:end);
        s1=strcat('dcal=args.Mom.',vname,loc,';');
        eval(s1);
    %     dcal=args.Mom.WeightPredict{1, 1}.w_i(1,1);
        error_delta=1e-10/dcal;%1e-5;%
        s2=strcat('args.',vname,loc,'=args.',vname,loc,'+error_delta;');
        eval(s2);
    %     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)+error_delta;
        [~,error1]=ae_ff(train_data,train_label,args);
        s3=strcat('args.',vname,loc,'=args.',vname,loc,'-2*error_delta;');
        eval(s3);
    %     args.WeightPredict{1, 1}.w_i(1,1)=args.WeightPredict{1, 1}.w_i(1,1)-2*error_delta;
        [~,error2]=ae_ff(train_data,train_label,args);
        dreal=(error1-error2)/error_delta/2;
        accuracy=abs((dcal-dreal)/dreal);
end

% figure;
% plot((1:length(args.Er))*100,args.Er);
% title('');
% xlabel('ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½');
% ylabel('ï¿½ï¿½ï¿?);

i1=1;
data=test_data(i1);
label=test_label(i1);
T=size(data{1},1);
L=size(label{1},1);
[predict,error2]=ae_ff(data,label,args);

figure;
plot(1:T,data{1},'b.',T:-1:T-L+1,label{1},'r.',T:-1:T-L+1,predict,'g.');
%}

%{
load 'args_fsc_No3_1231.mat';
load('fscData.mat');

label=hotWindPressLabel;
tic
[predict,Er]=fsc_ff({dHWP},{[hotWindPressLabel,~hotWindPressLabel]},args);
toc
data=hotWindPress(1:end-delay,1);
range1=hotWindPressLabel(delay+1:end,1);

% load('fscDataHWP.mat');
% label=[false(delay,1);labelTrain(1:end-delay,1)];
% [predict,Er]=fsc_ff({dataTrain},{[label,~label]},args);
% data=data0Train(1:end-delay,1);

range1=label(delay+1:end,1);
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