%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号

opt=struct(...
    'date_str_begin','2013-01-01', ... %开始时间
    'date_str_end','2013-01-17', ...   %结束时间
    'len',360*24, ...%计算PCA所用时长范围
    'step',360*24 ...
    );  

load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));
load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
data0=data0(:,commenDim{GL(i1)});% 选取共有变量
M0=mean(data0(~sv,:));%除去换炉扰动后的均值方差
S0=std(data0(~sv,:),0,1);

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);           % end index

disp('begin to calculate PCA');
loc=0:opt.step:(eIndex-sIndex);
T=length(loc);
T2=zeros(T,1);
SPE=zeros(T,1);
numDims=size(data0,2);
pH=zeros(numDims,numDims,T);
eH=zeros(numDims,T);
tic;
for i1=1:length(loc)
    t1=sIndex+loc(i1)-opt.len+1;
    t2=sIndex+loc(i1);
    data1=data0(t1:t2,:);
    date1=date0(t1:t2,:);
    sv1=sv(t1:t2,:);
    M1=mean(data1(~sv1,:));%除去换炉扰动
    S1=std(data1(~sv1,:),0,1);
    data_st=(data1-ones(size(data1,1),1)*M1)./(ones(size(data1,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1)=P;
    eH(:,i1)=E;
    [T2(i1,1),SPE(i1,1)]=pca_indicater(data_st(end,:),P,E,7);
end
toc;
%% 矩阵相似度分析
% p=pH(:,:,2000)/pH(:,:,1);
% imshow(p/norm(p));
%{
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
% figure;
% T=size(data1,1);
% range2=(1:T)/360/24;
% for i1=1:6
%     subplot(3,2,i1);
%     plot(range2,data1(:,ipt(i1)));
%     title(commenVar{ipt(i1)});
% end

