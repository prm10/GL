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
% normal_class={
%     '2012-03-21 20:49','2012-03-22 16:25'
%     '2012-03-29 08:07','2012-03-30 03:43'
%     '2012-11-12 23:33','2012-11-14 17:32'
%     '2013-01-14 07:24','2013-01-15 14:07'
%     '2013-01-22 05:40','2013-01-25 00:37'
%     '2013-02-12 23:23','2013-02-13 08:22'
%     '2013-02-15 00:00','2013-02-19 23:57'
%     '2013-03-02 09:24','2013-03-06 03:43'
% };
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
normal_class={
    '2012-03-21 20:49','2012-03-22 16:25'
    '2013-01-14 07:24','2013-01-15 14:07'
    '2013-01-22 05:40','2013-01-25 00:37'
    '2013-03-02 09:24','2013-03-06 03:43'
    '2012-03-21 20:49','2012-03-22 16:25'
    '2012-03-29 08:07','2012-03-30 03:43'
    '2013-02-12 23:23','2013-02-13 08:22'
    '2013-02-15 00:00','2013-02-19 23:57'
};
fault_class={
    '2012-03-23 12:08:22','2'
    '2013-01-15 23:43:11','2'
    '2013-01-25 06:18:26','2'
    '2013-03-06 07:19:32','2'
	'2012-03-25 12:46:13','3'
    '2012-03-30 19:56:08','4'
    '2013-02-13 14:49:30','4'
    '2013-02-25 16:45:14','4'
};
num_case=size(normal_class,1);
%% 对每段正常数据建立一个pca模型
% nomal sample
%
model_P1=[];
model_E1=[];
model_M1=[];
model_S1=[];
for i1=1:size(normal_class,1)
    date_str_begin=normal_class{i1,1};
    date_str_end=normal_class{i1,2};
    sIndex=find(date0>datenum(date_str_begin),1);  % start index
    eIndex=find(date0>datenum(date_str_end),1);    % end index
    data1=data0(sIndex:eIndex,:);
    date1=date0(sIndex:eIndex,:);
    M1=mean(data1);
    S1=std(data1,0,1);
    data1_st=(data1-ones(size(data1,1),1)*M1)./(ones(size(data1,1),1)*S0);
    [P1,E1]=pca(data1_st);
    model_P1=cat(3,model_P1,P1);
    model_E1=cat(2,model_E1,E1);
    model_M1=cat(2,model_M1,M1');
    model_S1=cat(2,model_S1,S1');
end
%}
%% 对异常数据按窗口长度、步长等建立多个pca模型
%
%abnormal sample
model_P2=[];
model_E2=[];
model_M2=[];
model_S2=[];
model_D2=[];
model_C2=[];
model_A2=[];
hours=0.5;
minutes=60;
fault_hours=3;
for i1=1:size(fault_class,1)
    date_str_end=fault_class{i1,1};
    eIndex=find(date0>datenum(date_str_end),1);    % end index
    sIndex=eIndex-360*fault_hours;
    len=360*hours; %计算PCA所用时长范围
    step=6*minutes;
    data1=data0(sIndex-len:eIndex,:);
    date1=date0(sIndex-len:eIndex,:);
    M0=model_M1(:,i1)';
    [P1,E1,M1,S1,D1]=get_pca_models(data1,date1,len,step,M0,S0);
    model_P2=cat(3,model_P2,P1);
    model_E2=cat(2,model_E2,E1);
    model_M2=cat(2,model_M2,M1);
    model_S2=cat(2,model_S2,S1);
    model_D2=cat(1,model_D2,D1);
    model_C2=cat(1,model_C2,str2num(fault_class{i1,2})*ones(size(D1)));
    for i2=1:size(P1,3)
        model_A2=cat(3,model_A2,model_P1(:,:,i1)'*P1(:,:,i2));
    end
end
%}
%显示转换矩阵的图像
figure;
for i1=1:size(model_A2,3)
    subplot(size(model_A2,3)/num_case,num_case,i1);
    imagesc(abs(model_A2(:,:,i1)));
    title(fault_class{mod(i1-1,8)+1,2});
end
%% 计算相似度矩阵
%{
sim0=calculate_sim(model_P2,model_E2,5);
sim=sim0(:,:,3);
n=size(sim,1);
figure;
imagesc(sim);
axis equal;
axis([.5,n+.5,.5,n+.5]);
%}
