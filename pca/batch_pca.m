%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');
window_hours=48;
hours=2;
minutes=30;
opt=struct(...
    'date_str_begin','2013-03-04', ... %开始时间
    'date_str_end','2013-03-06 07:19:32', ...   %结束时间
    'len',360*hours, ...%计算PCA所用时长范围
    'step',6*minutes, ...%步长
    'window',max(window_hours,hours)*360 ...
    );

load(strcat(filepath,'data.mat'));
data0=data0(:,commenDim{GL(gl_no)});% 选取共有变量
% load(strcat('..\..\GL_data\',num2str(No(i1)),'\sv.mat'));
% sv=false(size(sv));
%% pca
%
disp('begin to calculate PCA');
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
S0=std(data0(normalState,:));
sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
loc=0:opt.step:(eIndex-sIndex);
T=length(loc);
T2=zeros(T,1);
SPE=zeros(T,1);
numDims=size(data0,2);
model_P=zeros(numDims,numDims,T);
model_E=zeros(numDims,T);
model_M=zeros(numDims,T);
model_S=zeros(numDims,T);
model_D=zeros(T,1);
ignore=0;
tic;
for i1=1:length(loc)
    t_window=sIndex+loc(i1)-opt.window+1;
    t1=sIndex+loc(i1)-opt.len+1;
    t2=sIndex+loc(i1);
    data1=data0(t1:t2,:);
    
    data_window=data0(t_window:t2,:);
    ns0=normalState(t_window:t2,:);
    data_window=data_window(ns0,:);
    M_window=mean(data_window);
    ns=normalState(t1:t2,:);
%     sv1=sv(t1:t2,:);
    data2=data1;         % no filter
    
%     data2=data2(~sv1,:); % remove stove change
%     ns=ns(~sv1,:);      
    
    data2=data2(ns,:);     % filter abnormal state
    
    if size(data2,1)/size(data1,1)<0.5
        disp(strcat('abnormal index: ',num2str(t1),':',num2str(t2)));
        disp(strcat('abnormal rate: ',num2str(size(data2,1)/size(data1,1))));
        ignore=ignore+1;
        continue;
    end

    M1=mean(data2);
    S1=std(data2,0,1);
    data_st=(data2-ones(size(data2,1),1)*M_window)./(ones(size(data2,1),1)*S0);
    [P,E]=pca(data_st);
    model_P(:,:,i1-ignore)=P;
    model_E(:,i1-ignore)=E;
    model_D(i1-ignore)=date0(t2);
    model_M(:,i1-ignore)=M_window;
    model_S(:,i1-ignore)=S1;
end
toc;

model_P=model_P(:,:,1:end-ignore);
model_E=model_E(:,1:end-ignore);
model_D=model_D(1:end-ignore);
model_M=model_M(:,1:end-ignore);
model_S=model_S(:,1:end-ignore);
disp(strcat('samples ignored: ',num2str(ignore)));
disp(strcat('models generated: ',num2str(length(model_D))));
%}
%% original data
%
data1=data0(sIndex+1:eIndex,:);
figure;
T=size(data1,1);
% range2=(1:T)/360/24;
range2=(1:T);
for i1=1:6
    subplot(3,2,i1);
    plot(range2,data1(:,ipt(i1)),(1:length(model_D))*opt.step,model_M(ipt(i1),:),'--');
    title(commenVar{ipt(i1)});
end
%}
%% 矩阵相似度分析
%
clear data0 date0;
disp('begin to calculate similarity');
tic;
n=size(model_P,3);
k=5;
sim0=zeros(n,n);
sim=zeros(n,n,k);
for i1=1:n-1
    for i2=i1+1:n
%         [r1,result]=simN(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),k);
        [r1,result]=simG(model_P(:,:,i1),model_P(:,:,i2),model_E(:,i1),model_E(:,i2),k);
        sim(i1,i2,:)=result;
        sim(i2,i1,:)=result;
        sim0(i1,i2)=r1;
        sim0(i2,i1)=r1;
    end
end
for i1=1:n
    sim(i1,i1,:)=1;
    sim0(i1,i1)=1;
end
toc;
save('..\..\GL_data\batch_pca.mat','sim','sim0','model_D');
%}
%% plot
batch=load('..\..\GL_data\batch_pca.mat');
% batch=load('..\..\GL_data\batch_pca_20121201_20130101_4h_20min_G.mat');
%{
%各个角度取平均
for i1=1:k
    sim(:,:,i1)=(sim(:,:,i1)-M_sim(i1))/S_sim(i1);
end
sim=mean(sim,3);

%}

for i1=1:k
%取单个角度
sim=real(batch.sim(:,:,i1));
% sim=(sim-min(min(sim)))/(max(max(sim))-min(min(sim)));%归一化
n=size(sim,1);
figure;
imagesc(sim);
axis equal;
axis([.5,n+.5,.5,n+.5]);
end

n=size(batch.sim0,1);
figure;
imagesc(batch.sim0);
axis equal;
axis([.5,n+.5,.5,n+.5]);
%% 聚类
%{
load('..\..\GL_data\batch_pca.mat');
W=sim-diag(diag(sim));
k=5;
sta=zeros(k,1);
C = SpectralClustering(W, k);
index=[];
for i1=1:k
    a=find(C==i1);
    index=[index;a];
    sta(i1)=length(a);
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
%}
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
