function sim=f_calculate_similarity(choice,p1,p2,e1,e2,L)
    cmd=strcat('sim=f_calculate_similarity_',num2str(choice),'(p1,p2,e1,e2,L);');    
    eval(cmd);
end


% 线性子空间主角度平方均值 S_{PCA}
function sim=f_calculate_similarity_1(p1,p2,e1,e2,L)
p=p1(:,1:L)'*(p2(:,1:L)*p2(:,1:L)')*p1(:,1:L);
sim=mean(diag(p));
end

% 规范化后的线性子空间主角度平方均值 S_{PCA}^{\lambda}
function sim=f_calculate_similarity_2(p1,p2,e1,e2,L)
w1=diag(e1).^0.5;
p=w1*p1'*(p2*diag(e2)*p2')*p1*w1;
% sim=sum(diag(p));% problem exists
sim=sum(diag(p))/(e1'*e2);% original
end

% 加权对比载荷矩阵 S_{PCA}^{PC-pair}
function sim=f_calculate_similarity_3(p1,p2,e1,e2,L)
sum_p=abs(sum((p1*diag(e1)).*(p2*diag(e2))));
sim=sum(sum_p)/(e1'*e2);
end

% % 规范化后的线性子空间主角度平方均值
% function sim=f_calculate_similarity_4(p1,p2,e1,e2,L)
% e1=e1/max(sqrt(sum(e1.^2)),1e-10);
% e2=e2/max(sqrt(sum(e2.^2)),1e-10);
% w1=diag(e1).^0.5;
% p=w1*p1'*(p2*diag(e2)*p2')*p1*w1;
% sim=sum(diag(p));
% end

% 规范化后的线性子空间主角度和 S_{PCA}^{normalized}
function sim=f_calculate_similarity_4(p1,p2,e1,e2,L)
e1=e1/sum(e1);
e2=e2/sum(e2);
w1=sqrt(diag(e1));
w2=sqrt(diag(e2));
p=w1*(p1'*p2)*w2;
[~,S,~]=svd(p);
sim=sum(diag(S));%/sqrt(sum(e1))/sqrt(sum(e2));
end

% 加权对比载荷矩阵距离 D_{PCA}
function sim=f_calculate_similarity_5(p1,p2,e1,e2,L)
e1=e1/sum(e1);
e2=e2/sum(e2);
sum_p=sum(sum(((p1*diag(e1))-(p2*diag(e2))).^2));
sim=sqrt(sum_p);
end

% 未规范化的线性子空间主角度平方均值 S_{PCA}^{\lambda,subspace}
function sim=f_calculate_similarity_6(p1,p2,e1,e2,L)
w1=diag(e1).^0.5;
p=w1*p1'*(p2*diag(e2)*p2')*p1*w1;
sim=sum(diag(p));% problem exists
% sim=sum(diag(p))/(e1'*e2);% original
end

% 规范化后的线性子空间主角度平方均值
% function sim=f_calculate_similarity_4(p1,p2,e1,e2,L)
% w1=diag(e1).^0.5;
% p=w1*p1'*(p2*diag(e2)*p2')*p1*w1;
% sim=sum(diag(p))/sqrt(sum(e1.^2)*sum(e2.^2));% advanced
% end