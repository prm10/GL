function MPCA_update_model()
%{
data_MPCA:   cell(C,1)
model_MPCA:  cell(C,6): M,S,P,E,spe_limit,t2_limit
%}
global data_MPCA model_MPCA k_PCs;
C=size(data_MPCA,1);
N=size(data_MPCA{1},2);%dimension
model_MPCA=cell(C,4);
for i1=1:C
    trainset=data_MPCA{i1};
    %是否需要每个样本点分别记录M和S？
    M=mean(trainset);
    S=std(trainset,0,1);
    trainset_st=(trainset-ones(size(trainset,1),1)*M)./(ones(size(trainset,1),1)*S);
    [P,E]=pca(trainset_st);
    theta1=sum(E(k_PCs+1:end));
    theta2=sum(E(k_PCs+1:end).^2);
    theta3=sum(E(k_PCs+1:end).^3);
    h0=1-2/3*theta1*theta3/theta2^2;
    c_a=3.7;%2.58;
    spe_limit=theta1*(c_a*h0*sqrt(2*theta2)/theta1+1+theta2*h0*(h0-1)/theta1^2).^(1/h0);
    F_a=2;
    t2_limit=(N-1)*(N+1)*k_PCs/N/(N-k_PCs)*F_a;
    model_MPCA{i1,1}=M;
    model_MPCA{i1,2}=S;
    model_MPCA{i1,3}=P;
    model_MPCA{i1,4}=E;
    model_MPCA{i1,5}=spe_limit;
    model_MPCA{i1,6}=t2_limit;
end
