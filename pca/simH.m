function [sim,sim_k]=simH(p1,p2,e1,e2,k)%,m1,m2,s1,s2)
% p2=(ones(size(p1,1),1)*(2*(sum(p1.*p2)>0)-1)).*p2;
e=e1+e2;
e=e/sum(e);
sum_p=abs(sum(p1.*p2));
sim_k=sum_p(1:k)';
sim=sum_p*e;
end