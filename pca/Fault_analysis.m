clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');
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
S0=std(data0(normalState,:));
%% sample
%{
{start time,end time, class}
 1正常；2悬料；3滑料；4管道；5作料；6炉凉
%}
normal_class={
    '2012-03-21 20:49','2012-03-22 16:25'
    '2012-03-29 08:07','2012-03-30 03:43'
    '2012-11-12 23:33','2012-11-14 17:32'
    '2013-01-14 07:24','2013-01-15 14:07'
    '2013-01-22 05:40','2013-01-25 00:37'
    '2013-02-12 23:23','2013-02-13 08:22'
    '2013-02-15 00:00','2013-02-19 23:57'
    '2013-03-02 09:24','2013-03-06 03:43'
};
% fault_class={
%     '2012-03-23 12:08:22','2'
%     '2012-03-25 12:46:13','3'
%     '2012-03-30 19:56:08','4'
%     '2013-01-15 23:43:11','2'
%     '2013-01-25 06:18:26','2'
%     '2013-02-13 14:49:30','4'
%     '2013-02-25 16:45:14','4'
%     '2013-03-06 07:19:32','2'
% };
fault_class={
    '2012-03-23 12:08:22','2'
    '2012-03-25 12:46:13','3'
    '2013-01-15 23:43:11','2'
    '2013-03-06 07:19:32','2'
    '2013-01-25 06:18:26','2'
    '2012-03-30 19:56:08','4'
    '2013-02-13 14:49:30','4'
    '2013-02-25 16:45:14','4'
};
%% 分别对正常和异常数据建模
model_P=[];
model_E=[];
model_M=[];
model_S=[];
model_D=[];
model_C=[];
% nomal sample
hours=2;
minutes=30;
for i1=1:size(normal_class,1)
    date_str_begin=normal_class{i1,1};
    date_str_end=normal_class{i1,2};
    sIndex=find(date0>datenum(date_str_begin),1);  % start index
    eIndex=find(date0>datenum(date_str_end),1);    % end index
    len=360*hours; %计算PCA所用时长范围
    step=6*minutes;
    data1=data0(sIndex-len:eIndex,:);
    date1=date0(sIndex-len:eIndex,:);
    [P1,E1,M1,S1,D1]=get_pca_models(data1,date1,len,step,S0);
    model_P=cat(3,model_P,P1);
    model_E=cat(2,model_E,E1);
    model_M=cat(2,model_M,M1);
    model_S=cat(2,model_S,S1);
    model_D=cat(1,model_D,D1);
    model_C=cat(1,model_C,ones(size(D1)));
end
%abnormal sample
minutes=1;
fault_hours=1;
for i1=1:size(fault_class,1)
    date_str_end=fault_class{i1,1};
    eIndex=find(date0>datenum(date_str_end),1);    % end index
    sIndex=eIndex-360*fault_hours;
    len=360*hours; %计算PCA所用时长范围
    step=6*minutes;
    data1=data0(sIndex-len:eIndex,:);
    date1=date0(sIndex-len:eIndex,:);
    [P1,E1,M1,S1,D1]=get_pca_models(data1,date1,len,step,S0);
    model_P=cat(3,model_P,P1);
    model_E=cat(2,model_E,E1);
    model_M=cat(2,model_M,M1);
    model_S=cat(2,model_S,S1);
    model_D=cat(1,model_D,D1);
    model_C=cat(1,model_C,str2num(fault_class{i1,2})*ones(size(D1)));
end
sim0=calculate_sim(model_P,model_E,5);
%% 计算相似度矩阵
sim=sim0(:,:,3);
n=size(sim,1);
figure;
imagesc(sim);
axis equal;
axis([.5,n+.5,.5,n+.5]);
