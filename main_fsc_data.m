clc;close all;clear;
%
No=[2,3,5];
GL=[7,1,5];
ipt=[1;8;13;17;20;24];
plotvariable;
i1=1;%高炉编号
load(strcat('data\',num2str(No(i1)),'\data_labeled.mat'));
i2=4;%:length(input0)
data1=input0{i2}(:,commenDim{GL(i1)});

i0=17;
hotWindPress=data1(:,i0);
% coldWind=data1(:,8);

% md=zeros(length(hotWindPress),1);
% sd=zeros(length(hotWindPress),1);
% for i1=1:length(hotWindPress)
%     index=(max(1,i1-720):i1);
%     tempData=hotWindPress(index);
%     md(i1)=median(tempData);
%     sd(i1)=std(tempData);
% end
% sHWP=(hotWindPress-md)./max(sd,0.0001);
% dHWP=hotWindPress(2:end,:)-hotWindPress(1:end-1,:);
% dHWP=[0;dHWP/std(dHWP)];

[indexChange,sHWP,dHWP]=FindStoveChange(hotWindPress);
%% 人工微调 i1=1; i2=6;
%{
indexChange(120:155,1)=1;
indexChange(360:378,1)=0;
indexChange(378:410,1)=1;
indexChange(645:670,1)=1;
indexChange(943:980,1)=1;
indexChange(1211:1245,1)=1;
indexChange(2680:2720,1)=1;
indexChange(3700:3730,1)=1;
%}

figure;
subplot(311);
plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');
subplot(312);
plot(find(~indexChange),dHWP(~indexChange),'b.',find(indexChange),dHWP(indexChange),'r.');
subplot(313);
plot(find(~indexChange),sHWP(~indexChange),'b.',find(indexChange),sHWP(indexChange),'r.');
% figure;
% plot(find(~indexChange),hotWindPress(~indexChange),'b.',find(indexChange),hotWindPress(indexChange),'r.');

delay=60;
% hotWindPressLabel=[false(delay,1);indexChange(1:end-delay,1)];
indexTrain=1:2e4;
indexTest=2e4+1:length(hotWindPress);

data0Train=hotWindPress(indexTrain);
dataTrain=dHWP(indexTrain);
labelTrain=indexChange(indexTrain);

data0Test=hotWindPress(indexTest);
dataTest=dHWP(indexTest);
labelTest=indexChange(indexTest);
save('fscDataHWP.mat','data0Train','dataTrain','labelTrain'...
    ,'data0Test','dataTest','labelTest','delay');

csvwrite('fscDataHWP.csv',[hotWindPress,dHWP,indexChange]);
%}