function pH=directionUnify(pH)
for i1=2:size(pH,3)
    pH(:,:,i1)=(ones(size(pH,1),1)*(2*(sum(pH(:,:,i1).*pH(:,:,i1-1))>0)-1)).*pH(:,:,i1);
end
