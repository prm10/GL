%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');
hours=24;
minutes=10;
opt=struct(...
    'date_str_begin','2012-10-01', ... %开始时间
    'date_str_end','2012-10-21', ...   %结束时间
    'len',360*hours, ...%计算PCA所用时长范围
    'step',6*minutes ...
    );

load(strcat(filepath,'data.mat'));
data0=data0(:,commenDim{GL(gl_no)});% 选取共有变量
% load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
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

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
%% pca
%
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
%     sv1=sv(t1:t2,:);
    data2=data1;         % no filter
    
%     data2=data2(~sv1,:); % remove stove change
%     ns=ns(~sv1,:);      
    
    data2=data2(ns,:);     % filter abnormal state
    
    if size(data2,1)/size(data1,1)<0.5
        disp(strcat('abnormal index: ',num2str(t1),':',num2str(t2)));
        disp(strcat('abnormal rate: ',num2str(size(data2,1)/size(data1,1))));
        continue;
    end

    M1=mean(data2);
    S1=std(data2,0,1);
    data_st=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1)=P;
    eH(:,i1)=E;
    [T2(i1,1),SPE(i1,1)]=pca_indicater(data_st(end,:),P,E,3);
end
toc;
clear data0 date0;
%}
%% 矩阵相似度分析
%
tic;
n=size(pH,3);
sim=zeros(n,n);
for i1=1:n
    for i2=i1+1:n
        result=simH(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),5);
        sim(i1,i2)=result;
        sim(i2,i1)=result;
        
%         e1=eH(:,i1);
%         e2=eH(:,i2);
%         if(sum(e1)==0 || sum(e2)==0)
%             continue;
%         end
%         [~,result]=simG(pH(:,:,i1),pH(:,:,i2),e1,e2,5);
%         sim(i1,i2)=result(2);
%         sim(i2,i1)=result(2);
    end
end
toc;
for i1=1:n
    sim(i1,i1,:)=1;
end
figure;
imagesc(sim);
axis equal;
axis([.5,n+.5,.5,n+.5]);
save('..\..\GL_data\batch_pca.mat','sim','D');
%}
%% 聚类
load('..\..\GL_data\batch_pca.mat');
W=sim-diag(diag(sim));
k=5;
C = SpectralClustering(W, k);
index=[];
for i1=1:k
    index=[index;find(C==i1)];
end
sim2=zeros(size(sim));
for i1=1:size(sim,1)
    for i2=1:size(sim,2)
        sim2(i1,i2)=sim(index(i1),index(i2));
    end
end
n=size(sim2,1);
figure;
imagesc(sim2);
axis equal;
axis([.5,n+.5,.5,n+.5]);
%% 特征分析
% place=[445,450,455,460];
% for i1=place
%     figure;
%     hist(sim(i1,:),20);
% end

% data_sim=sim(:,2500);
% [p,ci]=betafit(min(data_sim,0.99*ones(size(data_sim))),0.01);
% x=0:1e-3:1;
% figure;
% hold on;
% hist(data_sim,200);
% plot(x,size(data_sim,1)/200*x.^(p(1)-1).*(1-x).^(p(2)-1)./beta(p(1),p(2)));

% mean_sim=mean(sim);
% figure;
% % subplot(211);
% % plot(median(sim));
% % subplot(212);
% plot(mean_sim,'-*');
% datestr(D(2240))

%% Beta分布参数估计
%{
sIndex=find(D>datenum('2013-01-23'),1);  % start index
eIndex=find(D>datenum('2013-01-26'),1);  % end index
p_beta=zeros(eIndex-sIndex+1,2);
ci_beta=zeros(2*(eIndex-sIndex+1),2);
for i1=1:eIndex-sIndex+1
    data_sim=sim(:,i1+sIndex-1);
    if mean(data_sim)<0.3
        continue;
    end
    [p_beta(i1,:),ci_beta(2*i1-1:2*i1,:)]=betafit(min(data_sim,0.99*ones(size(data_sim))));
end
figure;
plot(p_beta);
%}
%% 
%{
day=63;
range=sIndex+day*opt.step-1:sIndex+day*opt.step-1+opt.len;
figure;
for i1=1:6
    subplot(3,2,i1);
    plot(data0(range,ipt(i1)));
%     plot(data1(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end
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
