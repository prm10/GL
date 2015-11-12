function area=area_in_a_big_circle(n,r1,r2,o2,flag)
% n��ÿ��СԲ����
% r1��СԲ�뾶
% r2����Բ�뾶
% o2����ԲԲ��λ��
% flag��0����ϡ�����У�1�����������
area=0;
figure;
hold on;
for i2=1:n
    if flag==0% flag��0����ϡ������
        offset=[0,2*r1*(i2-1)];
    else% flag��1�����������
        offset=[r1*(i2-1),sqrt(3)*r1*(i2-1)];
    end
    for i1=1:n
        o1=[r1+2*r1*(i1-1),r1]+offset;
        plot_circle(o1,r1);
        [area1,x,y]=area_between_two_circle(o1,r1,o2,r2);
        area=area+area1;
        plot(x,y,'.'); 
    end
end
% plot_circle(o2,r2);
axis equal;

function [area,x,y]=area_between_two_circle(o1,r1,o2,r2)
%r1ΪСԲ��r2Ϊ��Բ
l=sqrt(sum((o1-o2).^2));
if(r1+r2<=l)%����Բ���ཻ
    area=0;
    x=[];
    y=[];
else if(r2-r1>=l)%��Բ����СԲ
        area=pi*r1^2;
        [x1,y1]=meshgrid(o1(1)-r1:(2*r1/100):o1(1)+r1,o1(2)-r1:(2*r1/100):o1(2)+r1);
        a=((x1-o1(1)).^2+(y1-o1(2)).^2<r1^2);
        x=x1(a);
        y=y1(a);
    else%����Բ�ཻ
        [x1,y1]=meshgrid(o1(1)-r1:(2*r1/100):o1(1)+r1,o1(2)-r1:(2*r1/100):o1(2)+r1);
        a=((x1-o1(1)).^2+(y1-o1(2)).^2<r1^2)&((x1-o2(1)).^2+(y1-o2(2)).^2<r2^2);
        area=sum(sum(a))/size(x1,1)/size(x1,2)*r1^2*4;
        x=x1(a);
        y=y1(a);
    end
end


function plot_circle(o,r)
alpha=0:pi/50:2*pi;
x=o(1)+r*cos(alpha); 
y=o(2)+r*sin(alpha); 
plot(x,y,'-') ;
