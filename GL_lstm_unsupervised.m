clc;close all;clear;
rng(3);
GL2={'������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','��ѹ4','����ѹ��','���ѹ��','���ѹ��2','ȫѹ��','�ȷ�ѹ��','�ȷ�ѹ��2','ʵ�ʷ���','����¶�','�ȷ��¶�','���¶���','��������','��������','���¶���','�����½���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
GL3={'������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ2','��ѹ3','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��','��Сʱʵ����ú��'};
GL5={'������','͸����ָ��','CO','H2','CO2','��׼����','��������','�������','�ķ綯��','¯��ú����','¯��ú��ָ��','����ȼ���¶�','��ѹ','��ѹ������','����ѹ��','���ѹ��','ȫѹ��','�ȷ�ѹ��','ʵ�ʷ���','�ȷ��¶�','���¶���','��������','��������','���¶���','����ϵ��','�ķ�ʪ��','�趨��ú��','��Сʱʵ����ú��'};


% load('K:\GL_data\3\data.mat');
load('K:\GL_data\3\data_normalized.mat');
% range=400000:length(date0);
% plot(date0(range),data1(range,22)/3);

% datestr(date0(1920468),'yyyy-mm-dd HH:MM:SS')
%% 3�Ÿ�¯��2012-07-06�����Ϊѵ������2013��Ϊ���Լ�
global train_data train_label test_data test_label;
train_index=400000:1920468;
test_index=1920468:size(data1,1)-1;
label_index=22;
train_data=data1(train_index,:);
test_data=data1(test_index,:);
train_label=data1(train_index+1,label_index)/3;
% train_label=max(train_label,-1.5*ones(size(train_label)));
% train_label=min(train_label,1.5*ones(size(train_label)));
test_label=data1(test_index+1,label_index)/3;
% test_label=max(test_label,-1.5*ones(size(test_label)));
% test_label=min(test_label,1.5*ones(size(test_label)));
clear date0 data1 train_index test_index;
% ��������
args.maxecho=100;
args.momentum=0.99;
args.labellength=8000;
args.circletimes=100;
args.learningrate=1e-3;
args.outputtype='tanh';
args.layer=[size(train_data,2) 30 size(train_label,2)];

args=LSTM_initial(args);
[args]=LSTM_train(args);