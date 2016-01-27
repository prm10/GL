function [P,te]=pca(x)
[P,S1,~] = svd(x'*x/(size(x,1)-1));
te=diag(S1);
end