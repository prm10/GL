function maxD=findMaxGradient(W)
name=fieldnames(W);
v=[];
for i1=1:length(name)
%     name{i1}
%     max(max(abs(W.(name{i1}))))
    v=[v;W.(name{i1})(:)];
end
maxD=max(max(abs(v)));