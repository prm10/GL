function sim=simG(p1,p2,e1,e2,k)%,m1,m2,s1,s2)
e=e1+e2;
e=e/sum(e);
p=p1(:,1:k)'*(p2(:,1:k)*p2(:,1:k)')*p1(:,1:k);
e=eig(p);
sim=sum(e)/k;
end