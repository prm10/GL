clc;close all;clear;
r1=(1764+35)/100;
r2=(701+35)/100;
r3=(3587+29)/100;
lowest=-4.9;

o1=[0,-3.38];
o2=[sqrt((r1-r2)^2-o1(2)^2),0];
o3=[0,lowest+r3];

h=1e-1;
x1m=o2(1)*(1+r2/(r1-r2));
x1=0:h:x1m;
y1=o1(2)+sqrt(r1^2-x1.^2);

x21=x1(end);
y21=y1(end);
jiao2=atan(y21/(x21-o2(1)))-pi/4;
y22=r2*tan(jiao2);
y2=y21:-h:y22;
x2=o2(1)+sqrt(r2^2-y2.^2);

x3=0:h:14.4;
y3=o3(2)-sqrt(r3^2-x3.^2);

figure;hold on;
plot([o1(1) x1(end)],[o1(2) y1(end)]);
plot([o2(1) x2(end)],[o2(2) y2(end)]);
plot(x1,y1,'b',x2,y2,'r',x3,y3,'g',[x2(end) x3(end)],[y2(end) y3(end)],'y','linewidth',2);
plot(-x1,y1,'b',-x2,y2,'r',-x3,y3,'g',[-x2(end) -x3(end)],[y2(end) y3(end)],'y','linewidth',2);
grid;axis equal;

atan(x1(end)/(y1(end)-o1(2)))/pi*180