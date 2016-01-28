function simi=simiMat(pH,eH)
simi=zeros(size(pH,3),1);
for i1=2:size(pH,3)
%     temp=diag(pH(:,:,t1)/pH(:,:,t1-1));
%     simi(t1,1)=sum(1-temp(1:7));
    temp=pH(:,:,i1)-pH(:,:,i1-1);
    if ~isempty(find(sum(pH(:,:,i1).*pH(:,:,i1-1))<0, 1))
        disp(i1);
    end
    simi(i1,1)=sum(sum(temp(:,1:7).^2));%./(eH(1:7,t1))
end
