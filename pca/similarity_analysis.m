%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
i1=1;%高炉编号

% 'date_str_begin','2013-01-22', ... %开始时间
%     'date_str_end','2013-01-25 10:00:00', ...   %结束时间

hours=5;
minutes=10;
opt=struct(...
    'date_str_begin','2014-01-01', ... %开始时间
    'date_str_end','2014/1/5 19:38:57', ...   %结束时间
    'len',360*hours, ...%计算PCA所用时长范围
    'step',6*minutes ...
    );

load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));

data0=data0(:,commenDim{GL(i1)});% 选取共有变量

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
tic;
for i1=1:length(loc)
    t1=sIndex+loc(i1)-opt.len+1;
    t2=sIndex+loc(i1);
    data1=data0(t1:t2,:);
    D(i1)=date0(t2);
    ns=normalState(t1:t2,:);
    data2=data1;         % no filter   
    
%     data2=data2(ns,:);     % filter abnormal state
    
%     if size(data2,1)/size(data1,1)<0.5
% %         disp(strcat('abnormal index: ',num2str(t1),':',num2str(t2)));
% %         disp(strcat('abnormal rate: ',num2str(size(data2,1)/size(data1,1))));
%         continue;
%     end

    M1=mean(data2);
    S1=max(std(data2,0,1),1e-5*ones(1,size(data2,2)));
    data_st=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1)=P;
    eH(:,i1)=E;
%     [T2(i1,1),SPE(i1,1)]=pca_indicater(data_st(end,:),P,E,3);
end
toc;
clear data0 date0 normalState sv;
%% 矩阵相似度分析
model=load('..\..\GL_data\pca_model_24.mat');
tic;
m=size(model.pH,3);
n=size(pH,3);
k=5;
sim=zeros(m,n,k);
% for i1=1:n
%     result=simH(pH(:,:,i1),model.pH(:,:,:),eH(:,i1),eH(:,:));
%     sim(:,i1)=result;
% end

for i1=1:m
    for i2=1:n
%         [~,result]=simH(model.pH(:,:,i1),pH(:,:,i2),model.eH(:,i1),eH(:,i2),k);
        [~,result]=simG(model.pH(:,:,i1),pH(:,:,i2),model.eH(:,i1),eH(:,i2),k);
        sim(i1,i2,:)=result;
    end
end
toc;
% sim(1,end)=0;
figure;
imagesc(sim(:,:,1));
% axis equal;
% axis([.5,n+.5,.5,n+.5]);
%}
%% 看分布的差异
sta_m=zeros(n,k);
sta_s=zeros(n,k);
for i1=1:k
    sta_m(:,i1)=mean(sim(:,:,i1));
    sta_s(:,i1)=std(sim(:,:,i1));
    figure;
    subplot(211);
    plot(sta_m(:,i1));
    subplot(212);
    plot(sta_s(:,i1));
end

