clc;clear;close all;
No=[2,3,5,6];
GL=[7,1,5,6];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');

hours=24*5;
minutes=30;
opt=struct(...
    'date_str_begin','2013-02-24 23:57', ... %开始时间
    'date_str_end','2013-02-27 16:45:14', ...   %结束时间
    'len',360*hours, ...%计算PCA所用时长范围
    'step',6*minutes ...
    );


load(strcat(filepath,'data.mat'));
data0=data0(:,commenDim{GL(gl_no)});% 选取共有变量
sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
% datestr(date0(sIndex+6123))
%% original data
data1=data0(sIndex+1:eIndex,:);
date1=date0(sIndex+1:eIndex,:);
figure;
T=size(data1,1);
% range2=(1:T)/360/24;
range2=(1:T);
for i1=1:6
    subplot(3,2,i1);
    plot(date1,data1(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end