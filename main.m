clc;
clear all;

% data=load("ex1data1.txt");
T = readtable('covid_19_data_tr.csv');
death = T{:,3};
date = T{:,5};
maxvalue = max(death);
death = death/maxvalue;

X=death ;
y=date;
data=[X y];

plotData(X,y);
title("Covid-19 Death Rate in Turkey by scaled 10,000")
xlabel("day");
ylabel("death rate which is scaled by 10000")
legend('Training data', 'Linear regression')
set(gca, 'FontName','Times New Roman');
set(gca, 'Fontsize',8);

m = length(X);
X = [ones(m,1),data(:,1)]; 
theta = zeros(2, 1); 
iterations = 10000;
alpha = 0.01;

initialCostValue = computeCost(X, y, theta);
[theta J_history] = gradientDescent(X, y, theta, alpha, iterations);

hold on;
plot( X*theta,X(:,2), '-')
xlabel("day");
ylabel("death rate which is scaled by 10000")
legend('Training data', 'Linear regression')
set(gca, 'FontName','Times New Roman');
set(gca, 'Fontsize',8);

hold off 