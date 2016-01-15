clc;close all;clear;

No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;

range=1e4:2e4;
%
% load(strcat('..\data\',num2str(No(i1)),'\data_labeled.mat'));
% load(strcat('..\data\',num2str(No(i1)),'\sv.mat'));
load(strcat('../data/',num2str(No(i1)),'/data_labeled.mat'));
load(strcat('../data/',num2str(No(i1)),'/sv.mat'));
i2=6;
data1=input0{i2}(:,commenDim{GL(i1)});
sv1=sv{i2};

i3=17;
hotWindPress=data1(~sv1,i3);
Mh=M(i3);
Sh=S(i3);
T=size(hotWindPress,1);
hotWindPress=(hotWindPress-ones(T,1)*Mh)./(ones(T,1)*Sh);
dHWP=hotWindPress(2:end,:)-hotWindPress(1:end-1,:);
dHWP=[0;dHWP/std(dHWP)];
dHWP=max(dHWP,-ones(size(dHWP)));
dHWP=min(dHWP,ones(size(dHWP)));
%{
load(strcat('../../GL_data/',num2str(No(i1)),'/data.mat'));
load(strcat('../../GL_data/',num2str(No(i1)),'/sv.mat'));
% load(strcat('..\..\GL_data\',num2str(No(i1)),'\data.mat'));

data1=data0(range,commenDim{GL(i1)});
sv1=sv(range,:);
data1=data1(~sv1,:);

i3=17;
hotWindPress=data1(:,i3);
hotWindPress=smooth(hotWindPress);
clear data0 date0 data1 sv sv1;
%}


load('args_ae.mat');
disp('ff begin');
tic;
output=encode_ff(args,dHWP);
toc;

K=3;
Idx=kmeans(output,K);
str_cmd='plot(';
ind=cell(0);
figure;
for i1=1:K
    ind{i1}=find(Idx==i1);
    str_cmd=strcat(str_cmd,'ind{',num2str(i1),'},hotWindPress(ind{',num2str(i1),'},1),''.'',');
end
str_cmd=str_cmd(1:end-1);
str_cmd=strcat(str_cmd,');');
eval(str_cmd);
% plot(hotWindPress,'--');

