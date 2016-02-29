function [SPE,T2,t]=pca_indicater(y,P,te,m)
t=y*P(:,1:m);
T2=t.^2*(1./te(1:m));
% SPE=sum((y*(eye(size(y,2))-P(:,1:m)*P(:,1:m)')).^2,2);
SPE=sum((y*(P(:,m+1:end)*P(:,m+1:end)')).^2,2);
end