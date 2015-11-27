clc;close all;clear;
rng(3);
GL2={'富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','顶压4','富氧压力','冷风压力','冷风压力2','全压差','热风压力','热风压力2','实际风速','冷风温度','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','顶温下降管','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
GL3={'富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压2','顶压3','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','本小时实际喷煤量','上小时实际喷煤量'};
GL5={'富氧率','透气性指数','CO','H2','CO2','标准风速','富氧流量','冷风流量','鼓风动能','炉腹煤气量','炉腹煤气指数','理论燃烧温度','顶压','顶压西北上','富氧压力','冷风压力','全压差','热风压力','实际风速','热风温度','顶温东北','顶温西南','顶温西北','顶温东南','阻力系数','鼓风湿度','设定喷煤量','上小时实际喷煤量'};
% datestr(date0(1920468),'yyyy-mm-dd HH:MM:SS')
global train_data train_label test_data test_label way;
%% big unlabeled dataset 
% load('K:\GL_data\3\data_normalized.mat');
% way=0;% use big dataset
% % range=400000:length(date0);
% % plot(date0(range),data1(range,22)/3);
% delay=0;
% train_index=400000:1920468;
% test_index=1920468:size(data1,1)-delay;
% fea_index=[1:21,26,28];
% label_index=[22];
% train_data=data1(train_index,fea_index);
% test_data=data1(test_index,fea_index);
% train_label=data1(train_index+delay,label_index)/3;
% % train_label=max(train_label,-1.5*ones(size(train_label)));
% % train_label=min(train_label,1.5*ones(size(train_label)));
% test_label=data1(test_index+delay,label_index)/3;
% % test_label=max(test_label,-1.5*ones(size(test_label)));
% % test_label=min(test_label,1.5*ones(size(test_label)));
% clear date0 data1 train_index test_index;
%% small normal state dataset
load('K:\GL_data\3\data_labeled.mat');
way=1;%use small dataset
train_data=cell(0);
train_label=cell(0);
fea_index=[1:21,26,28];
label_index=[22];
for i1=1:length(label0)
    if(label0{i1}(1,1)==1) %normal state
        train_data=[train_data,input0{i1}(:,fea_index)];
        train_label=[train_label,input0{i1}(:,label_index)/3];
    end
end
test_data=train_data(1);
test_label=train_label(1);
%% 3号高炉，2012-07-06至年底为训练集；2013年为测试集
args_name='args.mat';
if(~exist('args_name','var'))
    % 参数设置
    args.maxecho=10;
    args.momentum=0.9;
    args.labellength=1000;
    args.circletimes=100;
    args.learningrate=1e1;
    args.outputtype='tanh';
    args.layer=[length(fea_index) 20 5 length(label_index)];
    args=LSTM_initial(args);
else
    load(args_name);
    % 改变些参数
    args.momentum=0.99;
    args.learningrate=1e-1;
    args.labellength=1000;
    args.maxecho=10;
    args.circletimes=100;
end

[args]=LSTM_train(args);
save('args.mat','args');
%% big dataset
% pos=400000:410000;
% [dout,error]=LSTM_ff({train_data(pos,:)},{train_label(pos,:)},args);
% figure;plot(pos,train_label(pos,:),pos,dout{1},'.');
% legend('actual','predict');
% figure;imshow(100*abs(args.Weight{1, 1}.w_i));
% figure;imshow(abs(args.Weight{1, 2}.w_k));
%% small dataset
[dout,error]=LSTM_ff(test_data,test_label,args);
figure;plot(1:length(test_label{1}),test_label{1},1:length(test_label{1}),dout{1},'.');
legend('actual','predict');
figure;imshow(100*abs(args.Weight{1, 1}.w_i));