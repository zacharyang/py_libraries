from sklearn.model_selection import cross_val_score


def cross_val_reg(model,name,X,y,scoring=None,folds=5):
    
    '''
        Takes a model, ie. instantiated class object, feature and target set and performs 
        5 fold cross validation using R^2 as a metric.
        
        Prints the scores, mean and std of the scores. 
        
    '''
    print( '------------------------------------\n')
    print('{}-Fold Cross Validated Results for Model: '.format(folds)+ name)
    
    # Compute cross val score # 
    
    scores = cross_val_score(model, X, y, cv=folds,scoring=scoring)
    
    if scoring==None:

        print('Performance Metric: R2')
    else:
        print('Performance Metric: '+scoring)

    print("Cross-validated scores:", scores)
    print("Mean score:", np.mean(scores))
    print('Std score:', np.std(scores))


def residual_plot(model,name,X,y,color):
    '''
        Takes a model, its name and 
    '''

    
    # Generate reisduals #
    preds=model.fit(X,y).predict(X)
    resids=y-preds

    
    # Plot reisduals vs y first # 
    
    ax=plt.scatter(preds,resids,alpha=0.2,color=color)
    
    # Add y=0 line # 
    uplim=max(ax.get_xlim())
    lowlim=min(ax.get_xlim())
    lines=np.linspace(lowlim,uplim)
    zeros=np.zeros(50)
    line=mlines.Line2D(lines,zeros,linestyle='--')
    ax.add_line(line)
    
    # Set title # 
    ax.set_title(name,fontsize=12)
    
    # Set labels # 
    ax.set_ylabel('Residuals')
    ax.set_xlabel('Predicted')
    
    
    plt.show()
    

def act_pred_plots(models,names,X,y,color):
    '''
        Takes a model, its name, feature set and predictor set and returns an actual vs predicted plot
    '''

    # Generate predictions #
    preds=model.fit(X,y).predict(X)
    
    # Plot reisduals vs y first # 

    ax=plt.scatter(preds,y,alpha=0.2,color=color)
    
    # Make it a square plot # 
    uplim=max(ax.get_xlim()+ax.get_ylim())
    lowlim=min(ax.get_xlim()+ax.get_ylim())
    ax.set_xlim(lowlim,uplim)
    ax.set_ylim(lowlim,uplim)
    
    # Add unit line # 
    lines=np.linspace(lowlim,uplim)
    line=mlines.Line2D(lines,lines,linestyle='--')
    ax.add_line(line)
    
    # Set title # 
    ax.set_title(name,fontsize=12)
    
    # Set axis labels # 

    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    
    plt.show()