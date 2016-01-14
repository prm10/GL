function simi=simiMat(pH)
for t1=2:size(pH,3)
%     temp=diag(pH(:,:,t1)/pH(:,:,t1-1));
%     simi(t1,1)=sum(1-temp(1:7));
    temp=pH(:,:,t1)-pH(:,:,t1-1);
    simi(t1,1)=norm(temp(1:7,1:7));
end
