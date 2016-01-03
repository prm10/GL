clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=2;%¸ßÂ¯±àºÅ
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
load 'args_fsc_No3_0103.mat';
delay=60;
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