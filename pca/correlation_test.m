%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
i1=2;%高炉编号

%     'date_str_begin','2013-02-20', ... %开始时间
%     'date_str_end','2013-02-26', ...   %结束时间

opt=struct(...
    'date_str_begin','2012-11-11', ... %开始时间
    'date_str_end','2012-11-17', ...   %结束时间
    'len',360*24*2, ...%计算PCA所用时长范围
    'wl',60, ...
    'step',1 ...
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


M0=mean(data0);
S0=std(data0,0,1);

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index

trainset=data0(sIndex-opt.len+1:sIndex,:);
M1=mean(trainset);
S1=std(trainset,0,1);
trainset_st=(trainset-ones(size(trainset,1),1)*M1)./(ones(size(trainset,1),1)*S1);
[P,E]=pca(trainset_st);

data1=data0(sIndex+1:eIndex,:);
data1_st=(data1-ones(size(data1,1),1)*M1)./(ones(size(data1,1),1)*S1);
k=2;
[spe,t_2,To]=pca_indicater(data1_st,P,E,k);
%% 
loc=opt.wl:opt.step:(eIndex-sIndex);
n=length(loc);
covT=zeros(k,k,n);
for i1=1:n
    t1=loc(i1)-opt.wl+1;
    t2=loc(i1);
    To2=To(t1:t2,:);
    covT(:,:,i1)=To2'*To2/(size(To2,1)-1);
end

ds=covT(1,2,:);
ds=ds(:);
figure;
plot(ds);
%%
%{
%% original data
figure;
T=size(data1,1);
range2=(1:T)/360/24;
for i1=1:6
    subplot(3,2,i1);
    plot(range2,data1(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end
%}
