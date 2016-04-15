%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');

opt=struct(...
    'date_str_begin','2012-03-22 16:25', ... %开始时间
    'date_str_end','2012-03-23 13:05:10', ...   %结束时间
    'len',360*24*1, ...%计算PCA所用时长范围
    'step',360*1 ...
    );

load(strcat(filepath,'data.mat'));
data0=data0(:,commenDim{GL(gl_no)});% 选取共有变量
% load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
% sv=(smooth(double(sv),30)>0.1);
% sv=false(size(sv));
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

% M0=mean(data0(~sv,:));%除去换炉扰动后的均值方差
% S0=std(data0(~sv,:),0,1);

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
disp('begin to calculate PCA');
loc=opt.step:opt.step:(eIndex-sIndex);
T=eIndex-sIndex;% from eIndex to sIndex
T2=zeros(T,1);
SPE=zeros(T,1);
T2_lim=zeros(T,1);
SPE_lim=zeros(T,1);
abnormal=false(T,1);

trainset=data0(sIndex-opt.len+1:sIndex,:);
