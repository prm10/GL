function [C,dis] = SpectralClustering(W, k)  
    [n,m] = size(W);   
    s = sum(W);  
    D = full(sparse(1:n, 1:n, s));  
    E = D^(-1/2)*W*D^(-1/2);  
    [Q, V] = eigs(E, k);  
    C = kmeans(Q, k);
    center=zeros(k);
    for i1=1:k
        center(:,i1)=mean(Q(C==i1,:));
    end
    dis=zeros(k);
    for i1=1:k
        for i2=1:k
            dis(i1,i2)=sqrt(sum((center(:,i1)-center(:,i2)).^2));
        end
    end
end 