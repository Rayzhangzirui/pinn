function [sc] = plot_train_sample(X, u, varargin    )
%
figure;
sc = scatter3(X(:,2),X(:,3),X(:,1),6, u, varargin{:});
end