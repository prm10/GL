function sim=calculate_sim(pH,eH,k)
n=size(pH,3);
sim=zeros(n,n,k+1);
for i1=1:n-1
    for i2=i1+1:n
%         [~,result]=simN(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),k);
        [s,result]=simG(pH(:,:,i1),pH(:,:,i2),eH(:,i1),eH(:,i2),k);
        sim(i1,i2,1:end-1)=result;
        sim(i2,i1,1:end-1)=result;
        sim(i1,i2,end)=s;
        sim(i2,i1,end)=s;
    end
end
for i1=1:n
    sim(i1,i1,:)=1;
end