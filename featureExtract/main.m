clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=3;
load(strcat('K:\GL_data\',num2str(No(i1)),'\data.mat'));

% load(strcat('..\data\',num2str(No(i1)),'\data_labeled.mat'));
% i2=4;%:length(input0)
% data1=input0{i2}(:,commenDim{GL(i1)});

formatOut = 'yyyy-mm-dd';
% 2012-11-16 09:39:00';'2012-11-16 10:29:08
date_str_begin=datestr([2012,12,14,00,00,00],formatOut);
date_str_end=datestr( [2012,12,17,00,00,00],formatOut);
range=find((date0>datenum(date_str_begin))&(date0<datenum(date_str_end)));
dimIndex=[8,13,17,20];

figure;
for i3=1:4
    subplot(2,2,i3);
    t=(date0(range,1)-date0(range(1),1))*24;
    plot(t,data0(range,commenDim{GL(i1)}(dimIndex(i3))));
%   title(commenVar{dimIndex(i3)});  
    title(ZCS{GL(i1),commenDim{GL(i1)}(dimIndex(i3))});
end
figure;
plot(t,data0(range,7));
title(commenVar{7});
