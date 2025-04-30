import numpy as np

def GD(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de gradiente

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   función que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                     nIter = gd_params[0] número de iteraciones
                     alpha = gd_params[1] tamaño de paso alpha

    f_params  :   lista de parametros para la funcion objetivo
                     kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                     X     = f_params['X'] Variable independiente
                     y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''

    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    Theta = []
    for t in range(nIter):
        p = grad(theta, f_params=f_params)
        theta = theta - alpha * p
        Theta.append(theta)
    return np.array(Theta)

def SGD(theta=[], grad=None, gd_params=[], f_params=[]):
    '''
    Descenso de gradiente estocástico

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente

    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter = gd_params['nIter'] número de iteraciones
                      alpha = gd_params['alpha'] tamaño de paso alpha
                      batch_size = gd_params['batch_size'] tamaño de la muestra

    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    (high, dim) = f_params['X'].shape
    batch_size = gd_params['batch_size']

    nIter = gd_params['nIter']
    alpha = gd_params['alpha']

    Theta = []
    for t in range(nIter):
        # Set of sampled indices
        smpIdx = np.random.randint(low=0, high=high, size=batch_size, dtype='int32')
        # sample
        smpX = f_params['X'][smpIdx]
        smpy = f_params['y'][smpIdx]
        # parametros de la funcion objetivo
        smpf_params = {'kappa': f_params['kappa'],
                       'X': smpX,
                       'y': smpy}

        p = grad(theta, f_params=smpf_params)
        theta = theta - alpha * p
        Theta.append(theta)

    return np.array(Theta)

def MGD(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de gradiente con momento (inercia)

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter = gd_params['nIter'] número de iteraciones
                      alpha = gd_params['alpha'] tamaño de paso alpha
                      eta   = gd_params['eta']  parametro de inercia (0,1]
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    eta = gd_params['eta']
    p_old = np.zeros(theta.shape)
    Theta = []
    for t in range(nIter):
        g = grad(theta, f_params=f_params)
        p = g + eta * p_old
        theta = theta - alpha * p
        p_old = p
        Theta.append(theta)
    return np.array(Theta)

def NAG(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso acelerado de Nesterov

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter = gd_params['nIter'] número de iteraciones
                      alpha = gd_params['alpha'] tamaño de paso alpha
                      eta   = gd_params['eta']  parametro de inercia (0,1]
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter = gd_params['nIter']
    alpha = gd_params['alpha']
    eta = gd_params['eta']
    p = np.zeros(theta.shape)
    Theta = []

    for t in range(nIter):
        pre_theta = theta - 2.0 * alpha * p
        g = grad(pre_theta, f_params=f_params)
        p = g + eta * p
        theta = theta - alpha * p
        Theta.append(theta)
    return np.array(Theta)

def ADADELTA(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de Gradiente Adaptable (ADADELTA)

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter    = gd_params['nIter'] número de iteraciones
                      alphaADA = gd_params['alphaADADELTA'] tamaño de paso alpha
                      eta      = gd_params['eta']  parametro adaptación del alpha
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    epsilon = 1e-8
    nIter = gd_params['nIter']
    alpha = gd_params['alphaADADELTA']
    eta = gd_params['eta']
    G = np.zeros(theta.shape)
    g = np.zeros(theta.shape)
    Theta = []
    for t in range(nIter):
        g = grad(theta, f_params=f_params)
        G = eta * g ** 2 + (1 - eta) * G
        p = 1.0 / (np.sqrt(G) + epsilon) * g
        theta = theta - alpha * p
        Theta.append(theta)
    return np.array(Theta)

def ADAM(theta=[], grad=None, gd_params={}, f_params={}):
    '''
    Descenso de Gradiente Adaptable con Momentum(A DAM)

    Parámetros
    -----------
    theta     :   condicion inicial
    grad      :   funcion que calcula el gradiente
    gd_params :   lista de parametros para el algoritmo de descenso,
                      nIter    = gd_params['nIter'] número de iteraciones
                      alphaADA = gd_params['alphaADAM'] tamaño de paso alpha
                      eta1     = gd_params['eta1'] factor de momentum para la direccion
                                 de descenso (0,1)
                      eta2     = gd_params['eta2'] factor de momentum para la el
                                 tamaño de paso (0,1)
    f_params  :   lista de parametros para la funcion objetivo,
                      kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                      X     = f_params['X'] Variable independiente
                      y     = f_params['y'] Variable dependiente

    Regresa
    -----------
    Theta     :   trayectoria de los parametros
                     Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    epsilon = 1e-8
    nIter = gd_params['nIter']
    alpha = gd_params['alphaADAM']
    eta1 = gd_params['eta1']
    eta2 = gd_params['eta2']
    p = np.zeros(theta.shape)
    v = 0.0
    Theta = []
    eta1_t = eta1
    eta2_t = eta2
    for t in range(nIter):
        g = grad(theta, f_params=f_params)
        p = eta1 * p + (1.0 - eta1) * g
        v = eta2 * v + (1.0 - eta2) * (g ** 2)
        # p = p/(1.-eta1_t)
        # v = v/(1.-eta2_t)
        theta = theta - alpha * p / (np.sqrt(v) + epsilon)
        eta1_t *= eta1
        eta2_t *= eta2
        Theta.append(theta)
    return np.array(Theta)