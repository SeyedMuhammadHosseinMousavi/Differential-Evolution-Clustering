function m=PlotRes(X, sol)
    % Cluster Centers
    m = sol.Position;
    k = size(m,1);
    % Cluster Indices
    ind = sol.Out.ind;    
    Colors = hsv(k);
    for j=1:k
        Xj = X(ind==j,:);
               subplot(2,3,1)
        plot(Xj(:,1),Xj(:,2),'x','LineWidth',1,'Color',Colors(j,:));title('DE');
        hold on;
%         plot(m(:,1),m(:,2),'ok','LineWidth',2,'MarkerSize',6);
                subplot(2,3,2)
        plot(Xj(:,1),Xj(:,3),'x','LineWidth',1,'Color',Colors(j,:));title('DE');
        hold on;
%         plot(m(:,1),m(:,3),'ok','LineWidth',2,'MarkerSize',6);
                subplot(2,3,3)
        plot(Xj(:,1),Xj(:,4),'x','LineWidth',1,'Color',Colors(j,:));title('DE');
        hold on;
%         plot(m(:,1),m(:,4),'ok','LineWidth',2,'MarkerSize',6);
                subplot(2,3,4)
        plot(Xj(:,2),Xj(:,3),'x','LineWidth',1,'Color',Colors(j,:));title('DE');
        hold on;
%         plot(m(:,2),m(:,3),'ok','LineWidth',2,'MarkerSize',6);
                subplot(2,3,5)
        plot(Xj(:,2),Xj(:,4),'x','LineWidth',1,'Color',Colors(j,:));title('DE');
        hold on;
%         plot(m(:,2),m(:,4),'ok','LineWidth',2,'MarkerSize',6);
                subplot(2,3,6)
        plot(Xj(:,3),Xj(:,4),'x','LineWidth',1,'Color',Colors(j,:));title('DE');
        hold on;
%         plot(m(:,3),m(:,4),'ok','LineWidth',2,'MarkerSize',6);
        
    end  
hold off;
end