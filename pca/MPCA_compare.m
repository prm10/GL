function [c,s]=MPCA_compare(P,E)
%{
model_MPCA:  cell(C,6): M,S,P,E,spe_limit,t2_limit
P: PCA's loading matrix of the current time
c: which class belong, from 1~C, if belong to none of them, return -1
s: similarity between P and Multiple Model
%}
global MPCA_model k_PCs;
C=size(MPCA_model,1);
sim=zeros(C,1);
for i1=1:C
    [sim(i1,1),eig1]=simG(P,MPCA_model{i1,3},E,MPCA_model{i1,4},k_PCs);
end
[s,c]=max(sim);

