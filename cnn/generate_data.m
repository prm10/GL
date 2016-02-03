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
data0=data0(s:end,:);
date0=date0(s:end,:);
sv=sv(s:end,:);
% plot(date0);
%% 正常状态
%{
17  热风压力<0.34
8   冷风流量<20
20  顶温东北>350
7   富氧流量<5000
%}
normalState=...
    data0(:,17)>0.34    ...
    & data0(:,8)>20     ...
    & data0(:,20)<350   ...
    & data0(:,7)>5000;

% M1=mean(data0(~sv,commenDim{GL(i1)}));%除去换炉扰动
% S1=std(data0(~sv,commenDim{GL(i1)}),0,1);

M1=mean(data0(normalState,commenDim{GL(i1)}));%除去异常炉况
S1=std(data0(normalState,commenDim{GL(i1)}),0,1);

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
minWidth=360*24*3;
index=normalArea(normalState,minWidth);
ind=2;
sIndex=index(ind,1);
eIndex=index(ind,2);

figure;
range=sIndex:eIndex;
for i1=1:6
    subplot(3,2,i1);
    plot(data0(range,ipt(i1)));%1e4:end
    title(commenVar{ipt(i1)});
end