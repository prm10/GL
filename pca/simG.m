function sim=simG(p1,p2,e1,e2)%,m1,m2,s1,s2)
e=e1+e2;
e=e/sum(e);
p=p1'*(p2*p2')*p1;
e=eig(p);
sum_p=abs(sum(p1.*p2));
sim=sum_p*e;
end