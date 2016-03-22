function sim=simN_fast(p1,ps2,e1,e2)%,m1,m2,s1,s2)
% p2=(ones(size(p1,1),1)*(2*(sum(p1.*p2)>0)-1)).*p2;
% e=e1+e2;
e=e1;
e=e/sum(e);
ps1=repmat(p1,[1,1,size(ps2,3)]);
sum_p=reshape(abs(sum(ps1.*ps2)),size(p1,1),size(ps2,3));
sim=e'*sum_p;
end