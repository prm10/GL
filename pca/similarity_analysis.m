%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号

len_trainset=360*24;
accu=1/24/6;
opt=struct(...
    'date_str_begin','2013-01-01', ... %开始时间
    'date_str_end','2013-02-01', ...   %结束时间
    'len',len_trainset, ...%计算PCA所用时长范围
    'step',ceil(len_trainset*accu) ...
    );

load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));
load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
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


M0=mean(data0(~sv,:));%除去换炉扰动后的均值方差
S0=std(data0(~sv,:),0,1);

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
% clear date0;
disp('begin to calculate PCA');
loc=0:opt.step:(eIndex-sIndex);
T=length(loc);
T2=zeros(T,1);
SPE=zeros(T,1);
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
    sv1=sv(t1:t2,:);
    data2=data1;         % no filter
    
    data2=data2(~sv1,:); % remove stove change
    ns=ns(~sv1,:);      
    
    data2=data2(ns,:);     % filter abnormal state
    
    if size(data2,1)/size(data1,1)<0.5
%         disp(strcat('abnormal index: ',num2str(t1),':',num2str(t2)));
%         disp(strcat('abnormal rate: ',num2str(size(data2,1)/size(data1,1))));
        continue;
    end

    M1=mean(data2);%除去换炉扰动
    S1=std(data2,0,1);
    data_st=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1)=P;
    eH(:,i1)=E;
    [T2(i1,1),SPE(i1,1)]=pca_indicater(data_st(end,:),P,E,3);
end
toc;
clear data0 date0 normalState sv;
%% 矩阵相似度分析
model=load('..\..\GL_data\pca_model.mat');
tic;
m=size(model.pH,3);
n=size(pH,3);
sim=zeros(m,n);
for i1=1:n
    result=simH(pH(:,:,i1),model.pH(:,:,:),eH(:,i1),eH(:,:));
    sim(:,i1)=result;
end

% for i1=1:n
%     for i2=i1:n
%         result=simH(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2));
% %         result=simG(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),3);
%         sim(i1,i2)=result;
%         sim(i2,i1)=result;
%     end
% end
toc;
% imshow(sim/max(max(sim)));
sim(1,end)=0;
figure;
imagesc(sim);
% axis equal;
% axis([.5,n+.5,.5,n+.5]);
%}

