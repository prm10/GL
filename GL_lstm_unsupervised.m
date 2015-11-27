clc;close all;clear;
rng(3);
GL2={'������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','��ѹ4','����ѹ��','���ѹ��','���ѹ��2','ȫѹ��','�ȷ�ѹ��','�ȷ�ѹ��2','ʵ�ʷ���','����¶�','�ȷ��¶�','���¶���','��������','��������','���¶���','�����½���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
GL3={'������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
GL5={'������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ������','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��'};
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
%% 3�Ÿ�¯��2012-07-06�����Ϊѵ������2013��Ϊ���Լ�
args_name='args.mat';
if(~exist('args_name','var'))
    % ��������
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
    % �ı�Щ����
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