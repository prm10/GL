function [P,E,M,S,D]=get_pca_models(data0,date0,len,step,S0)
%{
 ‰»Î
data0: (n0,m),n0=len+n*step
 ‰≥ˆ
P: m,m,n
E: m,n
M: m,n
S: m,n
D: n
%}
[n0,m]=size(data0);
loc=0:step:(n0-len);
n=length(loc);
P=zeros(m,m,n);
E=zeros(m,n);
M=zeros(m,n);
S=zeros(m,n);
D=zeros(n,1);
for i1=1:length(loc)
    t1=loc(i1)+1;
    t2=loc(i1)+len;
    data1=data0(t1:t2,:);
    M1=mean(data1);
    S1=std(data1,0,1);
    data_st=(data1-ones(size(data1,1),1)*M1)./(ones(size(data1,1),1)*S0);
    [p1,e1]=pca(data_st);
    P(:,:,i1)=p1;
    E(:,i1)=e1;
    M(:,i1)=M1;
    S(:,i1)=S1;
    D(i1)=date0(t2);
end