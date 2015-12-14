clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
% global train_data train_label test_data test_label;
% train_data=cell(0);
% train_label=cell(0);
% test_data=cell(0);
% test_label=cell(0);
% train_data=[train_data;x1(range1,:)];
% train_label=[train_label;x1(range2,:)];
% test_data=[test_data;x1(range1,:)];
% test_label=[test_label;x1(range2,:)];
i1=1;
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
i2=2;%:length(input0)
data=input0{i2}(:,commenDim{GL(i1)});
figure;%datestr(time0{2}(end),'yyyy-mm-dd HH:MM:SS')
for i1=1:6
    subplot(3,2,i1);
    plot(data(:,ipt(i1)));
    title(commenVar{ipt(i1)});
end
% [input,decode,predict]=GenerateData(data,lengthD,lengthP,indexD,indexP);
    
