%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号
%{
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
load(strcat('data\',num2str(No(i1)),'\sv.mat'));
i2=1;%1:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});
M1=M(:,commenDim{GL(i1)});
S1=S(:,commenDim{GL(i1)});
%删除换炉中的data
sv1=sv{i2};
data1=data1(sv1==0,:);
timeLen=360*1;
T=size(data1,1)-timeLen;
delay=60;
%}
load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));
load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
M1=mean(data0(~sv,commenDim{GL(i1)}));%除去换炉扰动
S1=std(data0(~sv,commenDim{GL(i1)}),0,1);
% '2013-01-15 23:43:11','2013-01-16 01:21:13','2'
date_str_begin='2013-01-14';
date_str_end='2013-01-17';
dayLen=1;   %计算高炉的稳态工作点所用时长/h
timeLen=360*1;  %计算PCA所用时长范围
sIndex=find(date0>datenum(date_str_begin)-dayLen,1);  % start index
trainIndex=find(date0>datenum(date_str_begin),1);     % train data start index
eIndex=find(date0>datenum(date_str_end),1);           % end index
T=eIndex-trainIndex;                                  % search data time
data1=data0(sIndex:eIndex,commenDim{GL(i1)});         % get commen variable of train data
sv1=sv(sIndex:eIndex,1);                              % stove change variable
timeLen2=trainIndex-sIndex;%训练集起点

disp('begin to calculate PCA');
step=6;
range=1:step:T;
T=length(range);
T2=zeros(T,1);
SPE=zeros(T,1);
numDims=size(data1,2);
pH=zeros(numDims,numDims,T);
eH=zeros(numDims,T);
tic;
for i1=1:length(range)
    t1=range(i1)+timeLen2;
%     range_avg=t1-timeLen2:t1-1;
%     data2=data1(range_avg,:);%用于计算均值M、方差S
%     sv2=sv1(range_avg,:);
%     M2=mean(data2(~sv2,:));%除去换炉扰动
%     S2=std(data2(~sv2,:),0,1);

    rang_pca=t1-timeLen:t1-1;
    data2=data1(rang_pca,:);%data for PCA
    sv2=sv1(rang_pca,:);
    data2=data2(~sv2,:);%除去换炉扰动
    data3=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data3);
    pH(:,:,i1)=P;
    eH(:,i1)=E;
    [T2(i1,1),SPE(i1,1)]=pca_indicater(data3(end,:),P,E,7);
end
toc;
%% 矩阵相似度分析
% p=pH(:,:,2000)/pH(:,:,1);
% imshow(p/norm(p));
%
T=size(T2,1);
range1=(1:T)*step/360/24;

pH=directionUnify(pH);
simi=simiMat(pH,eH);
figure;
plot(range1,simi);
%}
%% test
%{
i2=10;%1:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});
T=size(data1,1)-timeLen;
numDims=size(data1,2);
pH2=zeros(numDims,numDims,T);
for t1=1:T
    data2=data1(t1:t1+timeLen-1,:);
    M2=mean(data2);
    S2=std(data2,0,1);
    data3=(data2-ones(size(data2,1),1)*M2)./(ones(size(data2,1),1)*S2);
    [P,te]=pca(data3);
    pH2(:,:,t1)=P;
end
pH2=directionUnify(pH2);
simi=simiMat(pH2(:,:,1:60:end));
figure;
plot(simi);
%}
%% 画统计量
%{
% data3=(data1-ones(size(data1,1),1)*M2)./(ones(size(data1,1),1)*S2);
% [T2,SPE]=pca_indicater(data3,P,te,7);
figure;
T=size(T2,1);
range1=(1:T)*step/360/24;
subplot(211);
plot(range1,T2);
subplot(212);
plot(range1,SPE);
%}
figure;
T=size(data1,1);
range2=(1:T)/360/24;
for i1=1:6
    subplot(3,2,i1);
    plot(range2,data1(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end

