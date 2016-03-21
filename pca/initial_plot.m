clc;clear;close all;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%��¯���
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');
opt=struct(...
    'date_str_begin','2013-01-23 00:37', ... %��ʼʱ��
    'date_str_end','2013-01-25 09:54:05' ...   %����ʱ��
    );
load(strcat(filepath,'data.mat'));
data0=data0(:,commenDim{GL(gl_no)});% ѡȡ���б���
sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
data1=data0(sIndex:eIndex,:);
%% original data
figure;
T=size(data1,1);
range2=(1:T)/360/24;
for i1=1:6
    subplot(3,2,i1);
    plot(range2,data1(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end