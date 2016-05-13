function [sim,eig1]=simG(p1,p2,e1,e2,k)%,m1,m2,s1,s2)
% e=e1+e2;
% e=e/sum(e);

p=p1(:,1:k)'*(p2(:,1:k)*p2(:,1:k)')*p1(:,1:k);
% p=p1(:,1:k)'*p2(:,1:k);
eig1=sort(eig(p),'descend');

w1=diag(e1).^0.5;
p=w1*p1'*(p2*diag(e2)*p2')*p1*w1;
eig2=sort(eig(p),'descend');
sim=sum(eig2)/(e1'*e2);
end