clc;clear;close all;
No=[2,3,5,6];
GL=[7,1,5,6];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%高炉编号
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');
opt=struct(...
    'date_str_begin','2012-03-30', ... %开始时间
    'date_str_end','2012-03-31', ...   %结束时间
    'len',360*24*1, ...%计算PCA所用时长范围
    'step',360*1 ...
    );
load(strcat(filepath,'data.mat'));
% data0=medfilt1(data0,10);
data0=data0(:,commenDim{GL(gl_no)});% 选取共有变量
sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index

%% original data
data1=data0(sIndex+1:eIndex,:);
% data2=medfilt1(data1,10);
for i1=1:size(data1,2)
    data2(:,i1)=smooth(data1(:,i1),10);
end
figure;
T=size(data1,1);
% range2=(1:T)/360/24;
range2=(1:T);
for i1=1:6
    subplot(3,2,i1);
    plot(range2,data1(:,ipt(i1)),'--',range2,data2(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end