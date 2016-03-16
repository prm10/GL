%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号

hours=24;
minutes=30;
opt=struct(...
    'date_str_begin','2012-08-01', ... %开始时间
    'date_str_end','2013-03-01', ...   %结束时间
    'len',360*hours, ...%计算PCA所用时长范围
    'step',6*minutes ...
    );

%%
%{

load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));
% load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
data0=data0(:,commenDim{GL(i1)});% 选取共有变量
% sv=false(size(sv));%不去除换炉扰动
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

% M0=mean(data0);
% S0=std(data0,0,1);

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
% clear date0;
disp('begin to calculate PCA');
loc=0:opt.step:(eIndex-sIndex);
T=length(loc);
% T2=zeros(T,1);
% SPE=zeros(T,1);
D=zeros(T,1);
numDims=size(data0,2);
pH=zeros(numDims,numDims,T);
eH=zeros(numDims,T);
ignore=0;
tic;
for i1=1:length(loc)
    t1=sIndex+loc(i1)-opt.len+1;
    t2=sIndex+loc(i1);
    data1=data0(t1:t2,:);
    ns=normalState(t1:t2,:);
%     sv1=sv(t1:t2,:);
    data2=data1;         % no filter
    
%     data2=data2(~sv1,:); % remove stove change
%     ns=ns(~sv1,:);      
    
    data2=data2(ns,:);     % filter abnormal state
    
    if size(data1,1)-size(data2,1)>360
%         disp(strcat('abnormal index: ',num2str(t1),':',num2str(t2)));
%         disp(strcat('abnormal rate: ',num2str(size(data2,1)/size(data1,1))));
        ignore=ignore+1;
        continue;
    end

    M1=mean(data2);
    S1=std(data2,0,1);
    data_st=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1-ignore)=P;
    eH(:,i1-ignore)=E;
%     [T2(i1-ignore,1),SPE(i1-ignore,1)]=pca_indicater(data_st(end,:),P,E,3);
    D(i1-ignore)=date0(t2);
end
toc;
clear data0 date0;
pH=pH(:,:,1:end-ignore);
eH=eH(:,1:end-ignore);
D=D(1:end-ignore);
save(strcat('..\..\GL_data\pca_model_',num2str(hours),'.mat'),'pH','eH','D');
%}
%% similarity
load(strcat('..\..\GL_data\pca_model_',num2str(hours),'.mat'));
tic;
n=size(pH,3);
k=5;
sim=zeros(n,n,k);
for i1=1:n-1
    i1
    for i2=i1+1:n
%         result=simH(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2));
        [~,eig1]=simG(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),k);
        sim(i1,i2,:)=eig1;
        sim(i2,i1,:)=eig1;
    end
end
for i1=1:n
    sim(i1,i1,:)=1;
end
toc;
% sim(1,end)=0;
figure;
imagesc(sim(:,:,1));
axis equal;
axis([.5,n+.5,.5,n+.5]);
save(strcat('..\..\GL_data\sim_',num2str(hours),'.mat'),'sim');
%% 计算均值方差
%
load(strcat('..\..\GL_data\sim_',num2str(hours),'.mat'));
sim=reshape(sim,[],k);
M0=mean(sim);
S0=std(sim,0,1);