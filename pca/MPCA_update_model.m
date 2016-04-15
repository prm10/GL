function MPCA_update_model()
global data_MPCA;
C=size(data_MPCA,1);
N=size(data_MPCA{1},2);
for i1=1:C
    trainset=data_MPCA{i1};
    %是否需要分别记录M和S？
    M1=mean(trainset);
    S1=std(trainset,0,1);
    trainset_st=(trainset-ones(size(trainset,1),1)*M1)./(ones(size(trainset,1),1)*S1);
    [P,E]=pca(trainset_st);
%     pH(:,:,i1)=P;
%     eH(:,i1)=E;
    k=7;
    [spe,t_2]=pca_indicater(data1_st,P,E,k);
    F_a=2;
    t2_limit=(N-1)*(N+1)*k/N/(N-k)*F_a;
    theta1=sum(E(k+1:end));
    theta2=sum(E(k+1:end).^2);
    theta3=sum(E(k+1:end).^3);
    h0=1-2/3*theta1*theta3/theta2^2;
    c_a=3.7;%2.58;
    spe_limit=theta1*(c_a*h0*sqrt(2*theta2)/theta1+1+theta2*h0*(h0-1)/theta1^2).^(1/h0);
end
