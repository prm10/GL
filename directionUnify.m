function p1=directionUnify(p1)
p=p1(:,:,1);
for i1=2:size(p1,3)
    p1(:,:,i1)=(ones(size(p,1),1)*(2*(sum(p.*p1(:,:,i1))>0)-1)).*p1(:,:,i1);
end
