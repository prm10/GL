clc;close all;clear;

No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;%¸ßÂ¯±àºÅ
delay=60;

load(strcat('K:\GL_data\',num2str(No(i1)),'\data.mat'));
data1=data0(:,commenDim{GL(i1)});
i3=17;
hotWindPress=data1(:,i3);
hotWindPress=smooth(hotWindPress);
% md=zeros(length(hotWindPress),1);
% sd=zeros(length(hotWindPress),1);
% for i1=1:length(hotWindPress)
%     index=(max(1,i1-720):i1);
%     tempData=hotWindPress(index);
%     md(i1)=median(tempData);
%     sd(i1)=std(tempData);
% end
% sHWP=(hotWindPress-md)./max(sd,0.0001);
dHWP=hotWindPress(2:end,:)-hotWindPress(1:end-1,:);
dHWP=[0;dHWP/std(dHWP)];
% data=hotWindPress(1:end-delay,1);
clear data0 date0 data1;

load 'args_fsc_No3_0103.mat';
disp('begin to ff');
batches=100;
T=size(dHWP,1)-delay;
step=floor(T/batches);
sv=false(batches*step,1);
for i1=1:batches
    disp(strcat('batches: ',num2str(i1)))
    index=(i1-1)*step+1:i1*step+delay;
    [predict,~]=fsc_ff({dHWP(index,:)},{[false(step+delay,1),~false(step+delay,1)]},args);
    sv(index(1:step))=predict(delay+1:step+delay,1)>0.5;
end
data=hotWindPress(1:batches*step,1);
figure;
subplot(211);
plot(find(~sv),data(~sv),'b.',find(sv),data(sv),'r.');
subplot(212);
plot(predict(delay+1:end,1));

save(strcat('K:\GL_data\',num2str(No(i1)),'\sv.mat'),'sv');

%{
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
sv=cell(0);
for i2=1:length(input0)
    data1=input0{i2}(:,commenDim{GL(i1)});
    i3=17;
    hotWindPress=data1(:,i3);
    hotWindPress=smooth(hotWindPress);
    [hotWindPressLabel,sHWP,dHWP]=FindStoveChange(hotWindPress);

    label=hotWindPressLabel;
    [predict,Er]=fsc_ff({dHWP},{[hotWindPressLabel,~hotWindPressLabel]},args);
    data=hotWindPress(1:end-delay,1);

    % load('fscDataHWP.mat');
    % label=[false(delay,1);labelTrain(1:end-delay,1)];
    % [predict,Er]=fsc_ff({dataTrain},{[label,~label]},args);
    % data=data0Train(1:end-delay,1);

    range1=label(delay+1:end,1);
    predict_label=predict>0.5;
    range2=predict_label(delay+1:end,1);
    
    sv=[sv;range2];
    figure;
    subplot(311);
    plot(find(~range1),data(~range1),'b.',find(range1),data(range1),'r.');
    subplot(312);
    plot(find(~range2),data(~range2),'b.',find(range2),data(range2),'r.');
    subplot(313);
    plot(predict(delay+1:end,1));
end
save(strcat('data\',num2str(No(i1)),'\sv.mat'),'sv');
%}
