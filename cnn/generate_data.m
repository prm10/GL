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
data0=data0(s:end,commenDim{GL(i1)});
date0=date0(s:end,:);
sv=sv(s:end,:);

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
%% find one sample and then plot
minWidth=360*24*3;
index=normalArea(normalState,minWidth);
ind=1;
sIndex=index(ind,1);
eIndex=index(ind,2);
figure;
range=sIndex:eIndex;
for i1=1:6
    subplot(3,2,i1);
    plot(data1(range,ipt(i1)));%1e4:end
    title(commenVar{ipt(i1)});
end
% low_filter(data1(range,ipt(4)))

%% get data and target
len_win=6*20;
data_seg=data1(range,:);
data2=zeros(size(data_seg));
for i1=1:size(data_seg,1)-len_win
%     data2(i1,:)=std(data1(i1:i1+len_win-1,:),0,1);
    data2(i1,:)=median(data_seg(i1:i1+len_win-1,:));
end
figure;
for i1=1:6
    subplot(3,2,i1);
    plot(data2(:,ipt(i1)),'.');%1e4:end
    title(commenVar{ipt(i1)});
end

% csvwrite('data_cnn.csv',[hotWindPress,dHWP,indexChange]);