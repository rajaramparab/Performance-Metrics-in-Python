def confusion_matrix(y_true,y_pred):
    '''
    Parameters: 
        y_true: correct(Ground Truth) values of target in array

        y_pred: predicted values in array

    Returns:
        cmatrix: ndarray of shape (n_classes, n_classes)
        Confusion matrix.

    '''
    classes=set(y_true)
    n_classes=len(classes)
    #creating n_classesxn_classes empty matrix
    cmatrix=np.zeros((n_classes,n_classes),dtype=int)
    
    #updating matrix values by using index, matrix[[row index][colummn index]]
    # Predicted                      0(Negative)        1(Positive)
    #actual         0(Negative)   +1 if [0][0]-TN     +1 if [0][1]- FP
    #actual         1(Positive)   +1 if [1][0]- FN    +1 if [1][1]-TP
    
    for i in range(len(y_true)):
        cmatrix[y_true[i]][y_pred[i]] += 1
        
    return cmatrix
