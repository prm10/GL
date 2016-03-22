clc;close all;clear;
% datevec(ans)
batch=load('..\..\GL_data\batch_pca_20121201_20130101_2h_10min_G.mat');
%% 
%{
%{
%各个角度取平均
for i1=1:k
    sim(:,:,i1)=(sim(:,:,i1)-M_sim(i1))/S_sim(i1);
end
sim=mean(sim,3);

%}
n=size(batch.sim0,1);
figure;
imagesc(batch.sim0);
axis equal;
axis([.5,n+.5,.5,n+.5]);
%
for i1=1:5
%取单个角度
sim=batch.sim(:,:,i1);
sim=(sim-min(min(sim)))/(max(max(sim))-min(min(sim)));%归一化
n=size(sim,1);
figure;
imagesc(sim);
axis equal;
axis([.5,n+.5,.5,n+.5]);
end
%}
%% 提取日期
dv=datevec(batch.D);
duan=1:length(dv);
% duan=3215:4423;
% duan=2000:3200;
th=dv(duan,4);
tm=dv(duan,5);
% sim0=batch.sim0(duan,duan);
sim0=batch.sim(duan,duan,4);
sim_mean=mean(sim0);
%以小时为间隔
simh=zeros(24,1);
for i1=1:24
    simh(i1,1)=median(sim_mean(th==i1-1));
end
figure;
plot(0:23,simh,'-*');
%以分钟为间隔
simm=zeros(24*6);
