clc;close all;clear;

No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
load(strcat('data\',num2str(No(i1)),'\sv.mat'));
% load 'args_fsc_No3_0103.mat';
delay=60;

i2=1;%1:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});
M1=M(:,commenDim{GL(i1)});
S1=S(:,commenDim{GL(i1)});
%删除换炉中的data
sv1=sv{i2};
data1=data1(sv1==0,:);

timeLen=360*1;
T=size(data1,1)-timeLen;
numDims=size(data1,2);
pH=zeros(numDims,numDims,T);
for t1=1:T
% t1=1;
    data2=data1(t1:t1+timeLen-1,:);
    M2=mean(data2);
    S2=std(data2,0,1);
    data3=(data2-ones(size(data2,1),1)*M2)./(ones(size(data2,1),1)*S2);
    [P,te]=pca(data3);
    pH(:,:,t1)=P;
end
%% 矩阵相似度分析
% p=pH(:,:,2000)/pH(:,:,1);
% imshow(p/norm(p));
pH=directionUnify(pH);
simi=simiMat(pH(:,:,1:60:end));
figure;
plot(simi);

%% test
%
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

%画统计量
%{
data3=(data1-ones(size(data1,1),1)*M2)./(ones(size(data1,1),1)*S2);
[T2,SPE]=pca_indicater(data3,P,te,7);
figure;
subplot(211);
plot(T2);
subplot(212);
plot(SPE)
plot(T2);
figure;
for i1=1:6
    subplot(3,2,i1);
    plot(data1(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end
%}




