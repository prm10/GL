%% import data
clc;close all;clear;
No=[2,3,5];
GL=[7,1,5];
ipt=[7;8;13;17;20;24];
plotvariable;
gl_no=2;%��¯���
filepath=strcat('..\..\GL_data\',num2str(No(gl_no)),'\');
hours=24;
minutes=60;
opt=struct(...
    'date_str_begin','2012-09-01', ... %��ʼʱ��
    'date_str_end','2013-01-01', ...   %����ʱ��
    'len',360*hours, ...%����PCA����ʱ����Χ
    'step',6*minutes ...
    );

%%
%{
load(strcat(filepath,'data.mat'));
% load(strcat('..\..\GL_data\',num2str(No(gl_no)),'\sv.mat'));
data0=data0(:,commenDim{GL(gl_no)});% ѡȡ���б���
% sv=false(size(sv));%��ȥ����¯�Ŷ�
%{
17  �ȷ�ѹ��<0.34
8   �������<20
20  ���¶���>350
7   ��������<5000
%}
normalState=...
    data0(:,17)>0.32    ...
    & data0(:,8)>20     ...
    & data0(:,20)<450   ...
    & data0(:,7)>2000;

% M0=mean(data0(~sv,:));%��ȥ��¯�Ŷ���ľ�ֵ����
% S0=std(data0(~sv,:),0,1);

% M0=mean(data0);
% S0=std(data0,0,1);

sIndex=find(date0>datenum(opt.date_str_begin),1);  % start index
eIndex=find(date0>datenum(opt.date_str_end),1);    % end index
% clear date0;
disp('begin to calculate PCA');
loc=0:opt.step:(eIndex-sIndex);
T=length(loc);
% T2=zeros(T,1);
% SPE=zeros(T,1);
D=zeros(T,1);
numDims=size(data0,2);
pH=zeros(numDims,numDims,T);
eH=zeros(numDims,T);
ignore=0;
tic;
for i1=1:length(loc)
    t1=sIndex+loc(i1)-opt.len+1;
    t2=sIndex+loc(i1);
    data1=data0(t1:t2,:);
    ns=normalState(t1:t2,:);
%     sv1=sv(t1:t2,:);
    data2=data1;         % no filter
    
%     data2=data2(~sv1,:); % remove stove change
%     ns=ns(~sv1,:);      
    
    data2=data2(ns,:);     % filter abnormal state
    
    if size(data1,1)-size(data2,1)>60
%         disp(strcat('abnormal index: ',num2str(t1),':',num2str(t2)));
%         disp(strcat('abnormal rate: ',num2str(size(data2,1)/size(data1,1))));
        ignore=ignore+1;
        continue;
    end

    M1=mean(data2);
    S1=std(data2,0,1);
    data_st=(data2-ones(size(data2,1),1)*M1)./(ones(size(data2,1),1)*S1);
    [P,E]=pca(data_st);
    pH(:,:,i1-ignore)=P;
    eH(:,i1-ignore)=E;
%     [T2(i1-ignore,1),SPE(i1-ignore,1)]=pca_indicater(data_st(end,:),P,E,3);
    D(i1-ignore)=date0(t2);
end
toc;
clear data0 date0;
pH=pH(:,:,1:end-ignore);
eH=eH(:,1:end-ignore);
D=D(1:end-ignore);
disp(strcat('sample ignored: ',num2str(ignore)));
disp(strcat('model generated: ',num2str(length(D))));
save(strcat(filepath,'pca_model_',num2str(hours),'.mat'),'pH','eH','D');
%}
%% similarity
%{
load(strcat(filepath,'pca_model_',num2str(hours),'.mat'));
tic;
n=size(pH,3);
k=5;
sim=zeros(n,n,k);
for i1=1:n-1
    for i2=i1+1:n
%         [~,result]=simH(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),k);
        [~,result]=simG(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),k);
        sim(i1,i2,:)=result;
        sim(i2,i1,:)=result;
    end
end
for i1=1:n
    sim(i1,i1,:)=1;
end
toc;
% sim(1,end)=0;
% figure;
% imagesc(sim(:,:,4));
% axis equal;
% axis([.5,n+.5,.5,n+.5]);

% �����ֵ����
%
k=size(sim,3);
sim2=reshape(sim,[],k);
M_sim=mean(sim2);
S_sim=std(sim2,0,1);
save(strcat(filepath,'sim_',num2str(hours),'.mat'),'sim','M_sim','S_sim');
%}
%% ����
load(strcat(filepath,'pca_model_',num2str(hours),'.mat'));
load(strcat(filepath,'sim_',num2str(hours),'.mat'));
%{
%�����Ƕ�ȡƽ��
for i1=1:k
    sim(:,:,i1)=(sim(:,:,i1)-M_sim(i1))/S_sim(i1);
end
sim=mean(sim,3);

%}
%
%ȡ�����Ƕ�
sim=sim(:,:,2);

sim=(sim-min(min(sim)))/(max(max(sim))-min(min(sim)));%��һ��
n=size(sim,1);
figure;
imagesc(sim);
axis equal;
axis([.5,n+.5,.5,n+.5]);
%%
W=sim-diag(diag(sim));
k=100;
[C,dis]=SpectralClustering(W,k);
sta=zeros(k,1);

ind_c=cell(0);
for i1=1:k
    a=find(C==i1);
    ind_c{i1}=a;
    sta(i1)=length(a);
end

%��������������
[~,c_sort]=min(sum(dis));
while length(c_sort)<k
    %����ĳ��������c_sort���о����ֵ
    dis1=zeros(k,1);
    for i1=1:k
        for i2=1:length(c_sort)
            dis1(i1,1)=dis1(i1,1)+dis(i1,c_sort(i2));
        end
    end
    [~,ind_min]=sort(dis1);
    for i1=1:k
        if isempty(find(c_sort==ind_min(i1), 1))
            c_sort=[c_sort;ind_min(i1)];
            break;
        end
    end
end
index=[];
for i1=1:k
    index=[index;ind_c{c_sort(i1)}];
end

sim2=zeros(size(sim));
for i1=1:size(sim,1)
    for i2=1:size(sim,2)
        sim2(i1,i2)=sim(index(i1),index(i2));
    end
end
n=size(sim2,1);
figure;
imagesc(sim2);
axis equal;
axis([.5,n+.5,.5,n+.5]);