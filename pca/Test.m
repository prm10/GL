%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');

hours=3;
minutes=1;
opt=struct(...
    'date_str_begin','2013-01-23 00:37', ... %开始时间
    'date_str_end','2013-01-25 09:54:05', ...   %结束时间
    'len',360*hours, ...%计算PCA所用时长范围
    'step',6*minutes ...
    );

load(strcat(filepath,'data.mat'));
data0=data0(:,commenDim{GL(gl_no)});% 选取共有变量

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
%% pca
disp('begin to calculate PCA');
sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
loc=0:opt.step:(eIndex-sIndex);
T=length(loc);
% T2=zeros(T,1);
% SPE=zeros(T,1);
D=zeros(T,1);
numDims=size(data0,2);
pH=zeros(numDims,numDims,T);
eH=zeros(numDims,T);
tic;
for i1=1:length(loc)
    t1=sIndex+loc(i1)-opt.len+1;
    t2=sIndex+loc(i1);
    data1=data0(t1:t2,:);
    D(i1)=date0(t2);
    ns=normalState(t1:t2,:);
    data2=data1;         % no filter   
    M1=mean(data2);
    S1=max(std(data2,0,1),1e-5*ones(1,size(data2,2)));
    data_st=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1)=P;
    eH(:,i1)=E;
end
toc;
clear data0 date0 normalState sv;
%% 矩阵相似度分析
disp('begin to calculate similarity');
model=load(strcat(filepath,'pca_model_24.mat'));
load(strcat(filepath,'level_24.mat'));
level_limit=100;%取排名靠前的模型
a=level<=level_limit;
model.pH=model.pH(:,:,a);
model.eH=model.eH(:,a);
model.D=model.D(a);
tic;
m=size(model.pH,3);
n=size(pH,3);
k=5;
sim=zeros(m,n,k);
for i1=1:m
    for i2=1:n
%         [~,result]=simH(model.pH(:,:,i1),pH(:,:,i2),model.eH(:,i1),eH(:,i2),k);
        [~,result]=simG(model.pH(:,:,i1),pH(:,:,i2),model.eH(:,i1),eH(:,i2),k);
        sim(i1,i2,:)=result;
    end
end
toc;
figure;
imagesc(sim(:,:,1));
% axis equal;
% axis([.5,n+.5,.5,n+.5]);
%}
%% 看均值、方差、密度分布的差异
sta_m=zeros(n,k);
sta_s=zeros(n,k);
t=(1:n)*opt.step/360/24;
for i1=1:3
    sta_m(:,i1)=mean(sim(:,:,i1));
    sta_s(:,i1)=std(sim(:,:,i1));
    figure;
    subplot(211);
    plot(t,sta_m(:,i1));
    subplot(212);
    plot(t,sta_s(:,i1));
end

