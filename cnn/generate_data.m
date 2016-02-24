%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号
load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));
load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
s=25.2e4;
% 选择一段连续的公有信号
data0=data0(s:end,commenDim{GL(i1)});
date0=date0(s:end,:);
sv=sv(s:end,:);
% 去除换炉扰动
data0=data0(~sv,:);
date0=date0(~sv,:);

% ddata0=data0(2:end,:)-data0(1:end-1,:);
% plot(date0);
%% 正常状态
%{
17  热风压力<0.34
8   冷风流量<20
20  顶温东北>350
7   富氧流量<5000
%}
normalState=...
    data0(:,17)>0.32    ...
    & data0(:,8)>20     ...
    & data0(:,20)<450   ...
    & data0(:,7)>2000;

% M1=mean(data0(~sv,commenDim{GL(i1)}));%除去换炉扰动
% S1=std(data0(~sv,commenDim{GL(i1)}),0,1);

M1=mean(data0(normalState,:));%除去异常炉况
S1=std(data0(normalState,:),0,1);
data1=bsxfun(@rdivide,bsxfun(@minus,data0,M1),S1);

%{
date_str_begin='2013-01-14';
date_str_end='2013-01-17';
sIndex=find(date0>datenum(date_str_begin),1);  % start index
eIndex=find(date0>datenum(date_str_end),1);           % end index
figure;
range=sIndex:eIndex;
for i1=1:6
    subplot(3,2,i1);
    plot(data0(range,ipt(i1)));%1e4:end
    title(commenVar{ipt(i1)});
end
%}

%
%% find one sample and then plot
minWidth=360*24*3;
[index,ignore]=normalArea(normalState,minWidth);
ind=3;
sIndex=index(ind,1);
eIndex=index(ind,2);
ig=ignore{ind};
range=sIndex:eIndex;
figure;%yua
for i1=1:6
    subplot(3,2,i1);
    hold on;
    plot(data1(range,ipt(i1)),'.');%1e4:end
    plot(ig,data1(range(ig),ipt(i1)),'r*')
    title(commenVar{ipt(i1)});
end
% low_filter(data1(range,ipt(4)))

%% get data and target
len_data=360*5;% dataset cover over 5 hours
len_target=6*10;% target predict next 30 minutes
data_seg=data1(range,:);% concerned data
% sv_seg=sv(range,:);% stove change records of concerned data
target_std=zeros(size(data_seg,1)-len_data-len_target,size(data_seg,2));
target_mean=zeros(size(data_seg,1)-len_data-len_target,size(data_seg,2));
for i1=1:size(data_seg,1)-len_data-len_target
    data_target=data_seg(i1+len_data+1:i1+len_data+len_target,:);
    target_std(i1,:)=std(data_target,0,1);
    target_mean(i1,:)=median(data_target);
end
figure;
for i1=1:6
    subplot(3,2,i1);
    plot(target_mean(:,ipt(i1)),'.');%1e4:end    
    title(commenVar{ipt(i1)});
end
%}
%% 
%{
minWidth=360*24*3;% minimal normal state zone
index=normalArea(normalState,minWidth);
len_data=360*5;% dataset cover over 5 hours
len_target=6*60;% target predict next 30 minutes
csvPath='../../GL_data/cnn/';
for ind=1:size(index,1)
    ind
    range=index(ind,1):index(ind,2);
    data_seg=data1(range,:);% concerned data
    csvName=num2str(ind);
    data2csv(data_seg,len_data,len_target,csvPath,csvName)
end
%}
