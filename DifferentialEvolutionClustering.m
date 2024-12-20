%% Differential Evolution Clustering - (7 Jan 2022)
% Here, we are using evolutionary algorithms, in order to perform
% clustering task. The selected algorithm is Differential Evolution (DE)
% evolutionary computation as it is faster than others. We are using HTRU2
% dataset which is consisted of 17898 samples and 9 features. Features 5 to
% 8 are used in this code. All 4 selected features combinations are
% clustered and compared with K-Means clustering and Gaussian Mixture Model
% (GMM) Clustering visually and statistically. You can use your data and define
% your parameters. 'K' is the number of clusters, 'MaxIt' is number of
% iterations and 'nPop' is population size which are most important
% parameters.
% ------------------------------------------------
% System is using the following paper as dataset:
% R. J. Lyon, B. W. Stappers, S. Cooper, J. M. Brooke, J. D. Knowles, Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach, Monthly Notices of the Royal Astronomical Society 459 (1), 1104-1123, DOI: 10.1093/mnras/stw656
% This code is part of following projects, so please cite them to:
% Mousavi, Seyed Muhammad Hossein, S. Younes MiriNezhad, and Atiye Mirmoini. "A new support vector finder method, based on triangular calculations and K-means clustering." 2017 9th International Conference on Information and Knowledge Technology (IKT). IEEE, 2017.
% Mousavi, Seyed Muhammad Hossein, Vincent Charles, and Tatiana Gherman. "An Evolutionary Pentagon Support Vector Finder Method." Expert Systems with Applications 150 (2020): 113284.
% ------------------------------------------------
% Feel free to ontact me if you find any problem using the code:
% mosavi.a.i.buali@gmail.com
% SeyedMuhammadHosseinMousavi
% My Google Scholar: https://scholar.google.com/citations?user=PtvQvAQAAAAJ&hl=en
% My GitHub: https://github.com/SeyedMuhammadHosseinMousavi?tab=repositories 
% My ORCID: https://orcid.org/0000-0001-6906-2152
% My Scopus: https://www.scopus.com/authid/detail.uri?authorId=57193122985 
% My MathWorks: https://www.mathworks.com/matlabcentral/profile/authors/9763916#
% ------------------------------------------------
% Enjoy the code and wish me luck :)

%% Starting DE Clustering
clear;
warning('off');
% Loading
data = load('HTRU2.txt');
X = data;
%
k = 6; % Number of Clusters
%
CostFunction=@(m) ClusterCost(m, X);     % Cost Function
VarSize=[k size(X,2)];           % Decision Variables Matrix Size
nVar=prod(VarSize);              % Number of Decision Variables
VarMin= repmat(min(X),k,1);      % Lower Bound of Variables
VarMax= repmat(max(X),k,1);      % Upper Bound of Variables

% DE Parameters
%
MaxIt=40;       % Maximum Iterations
nPop=k*2;         % Population Size
%
beta_min=0.2;   % Lower Bound of Scaling Factor
beta_max=0.8;   % Upper Bound of Scaling Factor
pCR=0.2;        % Crossover Probability

% Start
empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.Out=[];
BestSol.Cost=inf;
pop=repmat(empty_individual,nPop,1);
for i=1:nPop
    pop(i).Position=unifrnd(VarMin,VarMax,VarSize);  
    [pop(i).Cost, pop(i).Out]=CostFunction(pop(i).Position);  
    if pop(i).Cost<BestSol.Cost
        BestSol=pop(i);
    end 
end
BestRes=zeros(MaxIt,1);

% DE Body
for it=1:MaxIt
    for i=1:nPop        
        x=pop(i).Position;        
        A=randperm(nPop);        
        A(A==i)=[];        
        a=A(1);
        b=A(2);
        c=A(3);       
        % Mutation
        beta=unifrnd(beta_min,beta_max,VarSize);
        y=pop(a).Position+beta.*(pop(b).Position-pop(c).Position);
        y=max(y,VarMin);
        y=min(y,VarMax);        
        % Crossover
        z=zeros(size(x));
        j0=randi([1 numel(x)]);
        for j=1:numel(x)
            if j==j0 || rand<=pCR
                z(j)=y(j);
            else
                z(j)=x(j);
            end
        end        
        NewSol.Position=z;
        [NewSol.Cost, NewSol.Out]=CostFunction(NewSol.Position);       
        if NewSol.Cost<pop(i).Cost
            pop(i)=NewSol;           
            if pop(i).Cost<BestSol.Cost
               BestSol=pop(i);
            end
        end
        
    end    
% Update Best Cost
BestRes(it)=BestSol.Cost;    
% Iteration 
disp(['In Iteration # ' num2str(it) ': Highest Cost IS = ' num2str(BestRes(it))]);    
% Plot 
DECenters=PlotRes(X, BestSol);
pause(0.01);
end
%
DElbl=BestSol.Out.ind;

% Plot DE Train
figure;
set(gcf, 'Position',  [600, 300, 600, 250])
plot(BestRes,':',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'MarkerEdgeColor','r',...
    'Color',[0.3,0.9,0.9]);
title('Differential Evolution Clustering Train')
xlabel('DE Iteration Number','FontSize',12,...
       'FontWeight','bold','Color','r');
ylabel('DE Best Cost Result','FontSize',12,...
       'FontWeight','bold','Color','r');
legend({'DE Train'});

%% K-Means Clustering for Comparison
[kidx,KCenters] = kmeans(X,k);
figure
set(gcf, 'Position',  [150, 50, 700, 400])
subplot(2,3,1)
gscatter(X(:,1),X(:,2),kidx);title('K-Means')
hold on;
plot(KCenters(:,1),KCenters(:,2),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,2)
gscatter(X(:,1),X(:,3),kidx);title('K-Means')
hold on;
plot(KCenters(:,1),KCenters(:,3),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,3)
gscatter(X(:,1),X(:,4),kidx);title('K-Means')
hold on;
plot(KCenters(:,1),KCenters(:,4),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,4)
gscatter(X(:,2),X(:,3),kidx);title('K-Means')
hold on;
plot(KCenters(:,2),KCenters(:,3),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,5)
gscatter(X(:,2),X(:,4),kidx);title('K-Means')
hold on;
plot(KCenters(:,2),KCenters(:,4),'ok','LineWidth',2,'MarkerSize',6);
subplot(2,3,6)
gscatter(X(:,3),X(:,4),kidx);title('K-Means')
hold on;
plot(KCenters(:,3),KCenters(:,4),'ok','LineWidth',2,'MarkerSize',6);
%
KMeanslbl=kidx;
%% Gaussian Mixture Model Clustering for Comparison
options = statset('Display','final'); 
gm = fitgmdist(X,k,'Options',options)
idx = cluster(gm,X);
figure
set(gcf, 'Position',  [50, 300, 700, 400])
subplot(2,3,1)
gscatter(X(:,1),X(:,2),idx);title('GMM')
hold on;
subplot(2,3,2)
gscatter(X(:,1),X(:,3),idx);title('GMM')
hold on;
subplot(2,3,3)
gscatter(X(:,1),X(:,4),idx);title('GMM')
hold on;
subplot(2,3,4)
gscatter(X(:,2),X(:,3),idx);title('GMM')
hold on;
subplot(2,3,5)
gscatter(X(:,2),X(:,4),idx);title('GMM')
hold on;
subplot(2,3,6)
gscatter(X(:,3),X(:,4),idx);title('GMM')
hold on;
%
GMMlbl=idx;

% MAE and MSE Errors
DE_GMM_MAE=mae(DElbl,GMMlbl);
DE_KMeans_MAE=mae(DElbl,KMeanslbl);
GMM_KMeans_MAE=mae(GMMlbl,KMeanslbl);
DE_GMM_MSE=mse(DElbl,GMMlbl);
DE_KMeans_MSE=mse(DElbl,KMeanslbl);
GMM_KMeans_MSE=mse(GMMlbl,KMeanslbl);
fprintf('DE vs GMM MAE =  %0.4f.\n',DE_GMM_MAE)
fprintf('DE vs K-Means MAE =  %0.4f.\n',DE_KMeans_MAE)
fprintf('GMM vs K-Means MAE =  %0.4f.\n',GMM_KMeans_MAE)
fprintf('DE vs GMM MSE =  %0.4f.\n',DE_GMM_MSE)
fprintf('DE vs K-Means MSE =  %0.4f.\n',DE_KMeans_MSE)
fprintf('GMM vs K-Means MSE =  %0.4f.\n',GMM_KMeans_MSE)
% fprintf('DE Centers vs K-Means Centers MSE =  %0.4f.\n',mse(DECenters,KCenters))
% fprintf('DE Centers vs K-Means Centers MAE =  %0.4f.\n',mae(DECenters,KCenters))



