close all;clear;clc;
% n��ÿ��СԲ����
% r1��СԲ�뾶
% r2����Բ�뾶
% o2����ԲԲ��λ��
% flag��0����ϡ�����У�1�����������
n=10;
r1=2;
r2=12;
o2=[27,15];
flag=1;
area1=area_in_a_big_circle(n,r1,r2,o2,flag);
flag=0;
area2=area_in_a_big_circle(n,r1,r2,o2,flag);