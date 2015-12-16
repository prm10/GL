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
i2=1;%:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});

%%
coldWind=data1(:,8);
md=zeros(length(coldWind),1);
sd=zeros(length(coldWind),1);
for i1=1:length(coldWind)
    index=(max(1,i1-360):i1);
    tempData=coldWind(index);
    md(i1)=median(tempData);
    sd(i1)=std(tempData);
end
flag=(coldWind-md)./sd;
figure;
subplot(211);
plot(coldWind,'.');
subplot(212);
plot(flag,'.');

% figure;%datestr(time0{2}(end),'yyyy-mm-dd HH:MM:SS')
% for i1=1:6
%     subplot(3,2,i1);
%     plot(data1(:,ipt(i1)));%1e4:end
%     title(commenVar{ipt(i1)});
% end


% [input,decode,predict]=GenerateData(data,lengthD,lengthP,indexD,indexP);

