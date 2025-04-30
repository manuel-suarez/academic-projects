import numpy as np
import scipy as sp

# Definimos algunas variables para acortar los nombres de funciones
SOLVE = np.linalg.solve
NORM = np.linalg.norm
CHOL = np.linalg.cholesky
RANK = np.linalg.matrix_rank
EIG  = sp.linalg.eig # To solve generalized eigenvalues problem
INV  = np.linalg.inv
QR   = sp.linalg.qr

# Función auxiliar para depurar (fallará si no es un ndarray)
def print_v(name, var, debug=True):
    if debug:
        print(f"{name} =", var, var.ndim, var.shape, np.isscalar(var))

def check_dims(var, ndim, m=None, n=None):
    assert var.ndim == ndim
    if ndim == 1:
        assert var.shape[0] == m
    if ndim == 2:
        assert var.shape[0] == m and var.shape[1] == n

# Algorithm 1: L-SR1 Trust-Region (L-SR1-TR) Method
# Erway, et. al., p. 8
def L_SR1_TR(theta_0=[], func=None, grad=None, gd_params={}, f_params={}):
    '''
        Stochastic SR1 Trust-region scheme (MB-LSR1)
        Griffin, et. al.
        A minibatch stochastic Quasi-Newton method adapted for nonconvex deep learning problems
        Algorithm 1, p. 5

        Parámetros
        -----------
        theta     :   condicion inicial (x0)
        fun       :   función de costo
        grad      :   gradiente de la función de costo
        gd_params :   lista de parametros para el algoritmo de descenso,
                        nIter    = gd_params['nIter'] número de iteraciones
                        nIter:            Max. num. of iterations
                        batch_size:         Batch size
                        mmr:                Memory size
                        delta_0:            Trust-region radius
                        eps:                Termination tolerance
                        gamma_0:
                        tau_1:
                        tau_2:
                        tau_3:
                        eta_1:
                        eta_2:
                        eta_3:
                        eta_4
                        alpha:              Step-size for Wolfe line-search (alpha = 1)
                        mu:                 (mu = 0.9)
        f_params  :   lista de parametros para la funcion objetivo,
                          kappa = f_params['kappa'] parametro de escala (rechazo de outliers)
                          X     = f_params['X'] Variable independiente
                          y     = f_params['y'] Variable dependiente

        Regresa
        -----------
        Theta     :   trayectoria de los parametros
                         Theta[-1] es el valor alcanzado en la ultima iteracion
    '''
    nIter       = gd_params['nIter']
    batch_size  = gd_params['batch_size']
    mem_size    = gd_params['mem_size']
    delta_0	    = gd_params['delta_0']
    eps	        = gd_params['eps']
    gamma_0	    = gd_params['gamma_0']
    #tau_1	    = gd_params['tau_1']
    #tau_2	    = gd_params['tau_2']
    #tau_3	    = gd_params['tau_3']
    #eta_1	    = gd_params['eta_1']
    #eta_2	    = gd_params['eta_2']
    #eta_3	    = gd_params['eta_3']
    #eta_4       = gd_params['eta_4']
    #alpha	    = gd_params['alpha']
    #mu	        = gd_params['mu']
    # Variables para el muestreo aleatorio de los lotes (batch size)
    (high, dim) = f_params['X'].shape
    # Inicialización de variables
    S = np.array([])
    Y = np.array([])
    Theta = []
    n = theta_0.shape[0] # Número de variables a optimizar, se usa para verificar el shape de los vectores y matrices generados
    print(f"Se están optimizando {n} variables")
    theta = np.array([theta_0]).T
    print_v("theta_0", theta, False)
    check_dims(theta, 2, n, 1)
    delta = delta_0
    gamma = gamma_0
    # g_k es el gradiente completo, gh_k es el gradiente de un lote seleccionado aleatoriamente
    e = 0 # Es el contador de cuantos renglones hemos eliminado de S y Y, es importante llevar la cuenta
    # ya que las dimensiones de los vectores y matrices que dependen de k deben considerar el efecto de reducción en e
    for iter in range(nIter):                         # Línea 2
        k = iter+1
        print(f"{30*'='}> Iteración {k}, n={n}, k={k}, e={e} <{30*'='}")
        # Muestreamos una lista con índices aleatorios para la conformación del lote
        sample_idxs = np.random.randint(low=0, high=high, size=batch_size, dtype='int32')
        # Obtenemos la muestra de observaciones y etiquetas
        sample_X = f_params['X'][sample_idxs]
        sample_y = f_params['y'][sample_idxs]
        # 1. Compute iteration batch I_k and g_k (added f_k)
        batch = {'kappa': f_params['kappa'], 'X': sample_X, 'y': sample_y}
        f = func(theta, f_params=batch)
        print_v("f", f, False)
        check_dims(f, 0)
        g = np.array([grad(theta, f_params=batch)]).T                 # Línea 1
        print_v("g", g, False)
        check_dims(g, 2, n, 1)
        norm_g = NORM(g)
        print_v("norm_g", norm_g, False)
        check_dims(norm_g, 0)

        # Trust-region subproblem (calc of p star with Algorithm 2)
        if iter == 0 or S.shape[0] == 0:                    # En la primera iteración o siempre que se haya hecho reset a la lista
            p = -delta * (g / norm_g)
            Bp = gamma * p
        else:
            p = TRsubproblem_solver_OBS(delta, gamma, g, Psi, Minv, n, k, e)         # Línea 7
            #print("Subproblem p =", p, p.shape, p.ndim)
            #print("Psi =", Psi, Psi.shape, Psi.ndim)
            #print("Minv =", Minv, Minv.shape, Minv.ndim)
            PsiTp = Psi.T.dot(p)
            #print("PsiTp =", PsiTp, PsiTp.shape, PsiTp.ndim)
            if Minv.ndim == 0:
                tmp = Psi * (PsiTp / Minv)
            else:
                tmp = Psi @ SOLVE(Minv, PsiTp)
            #print("tmp =", tmp, tmp.shape, tmp.ndim)
            Bp = gamma * p + tmp
        #print("Bp =", Bp, Bp.shape, Bp.ndim)
        print_v("p", p, False)
        check_dims(p, 2, n, 1)
        print_v("Bp", Bp, False)
        check_dims(Bp, 2, n, 1)

        Q_p = p.T.dot(g + 0.5*Bp)
        print_v("Q_p", Q_p, False)
        # Q_p debería ser un escalar pero es una matríz de 2 dimensiones de 1x1
        check_dims(Q_p, 2, 1, 1)
        norm_p = NORM(p)
        print_v("norm_p", norm_p, False)
        check_dims(norm_p, 0)
        # Compute new values of the function and gradient
        theta_new = theta + p                                               # Línea 10
        print_v("theta", theta, False)
        print_v("theta_new", theta_new, False)
        check_dims(theta_new, 2, n, 1)
        f_new = func(theta_new, f_params=batch)
        g_new = func(theta_new, f_params=batch)                             # Línea 11
        # Compute curvature pairs
        s = theta_new - theta # or assign p directly                        # Línea 11
        print_v("s", s, False)
        check_dims(s, 2, n, 1)
        y = g_new - g                                                       # Línea 11
        print_v("y", y, False)
        check_dims(y, 2, n, 1)

        # Compute the reduction-ratio rho = actual / estimated reduction
        ared = f_new - f
        pred = Q_p
        rho = ared/pred                                 # Línea 9
        print_v("rho", rho, False)
        # rho debería ser un escalar pero también es una matriz de 2 dimensiones de 1x1 (por las operaciones para calcularlo)
        check_dims(rho, 2, 1, 1)

        # Stop condition
        if norm_g < eps:                                # Línea 3
            break                                       # Línea 4

        # Update variables
        theta = theta_new
        Theta.append(theta)

        # Adjust TR radius
        if rho > 0.75:                                  # Línea 12
            if norm_p <= 0.8 * delta:
                delta_new = delta
            else:
                delta_new = 2 * delta
        else:
            if 0.1 <= rho and rho <= 0.75:
                delta_new = delta
            else:
                delta_new = 0.5 * delta
        delta = delta_new
        # delta es un tipo nativo de Python, debo analizar si conviene convertirlo a tipo escalar de Numpy
        #print_v("delta", delta)
        #check_dims(delta, 0)

        # Update conditions
        #print("Update conditions", iter)
        y_Bs = y - Bp
        print_v("y_Bs", y_Bs, False)
        check_dims(y_Bs, 2, n, 1)
        if np.abs(s.T.dot(y_Bs)) > 1e-8 * norm_p * NORM(y_Bs):
            if iter == 0:
                S = s
                Y = y
            else:
                try:
                    S = np.hstack((S, s))
                    Y = np.hstack((Y, y))
                except:
                    print("Error en hstack")
                    print_v("S", S)
                    check_dims(S, 2, n, k-e)
                    print_v("Y", Y)
                    check_dims(Y, 2, n, k-e)
                    raise
            # Removemos el primer elemento de acuerdo con el tamaño de la memoria
            if (S.shape[1] > mem_size):
                S = S[:, 1:]
                Y = Y[:, 1:]
            print_v("S", S)
            check_dims(S, 2, n, k-e)
            print_v("Y", Y)
            check_dims(Y, 2, n, k-e)
            # A partir de los vectores de curvatura construimos las matrices para el subproblema de la región de confianza
            #print(S.shape, S.ndim, len(S))
            while (S.shape[1] > 0):
                #print("Curvature pairs", S.shape[0])
                #print("S", S, S.ndim, S.shape)
                #print("Y", Y, Y.ndim, Y.shape)
                SY = S.T.dot(Y)
                print_v("SY", SY)
                check_dims(SY, 2, k-e, k-e)
                #print("SY", SY, SY.ndim, SY.shape)
                SS = S.T.dot(S)
                print_v("SS", SS)
                check_dims(SS, 2, k-e, k-e)
                #print("SS", SS, SS.ndim, SS.shape)
                if SY.ndim == 0:
                    LDLt = SY
                    eig_val = SY
                    lambda_hat_min = SY
                else:
                    LDLt = np.tril(SY) + np.tril(SY,-1).T
                    eig_val = EIG(LDLt, SS)[0]
                    #print("eig_val =", eig_val)
                    #print("eig_val =", eig_val, eig_val.shape, eig_val.ndim)
                    lambda_hat_min = np.min(eig_val)
                #print("LDLt", LDLt)
                #print("eig_val", eig_val)
                print_v("LDLt", LDLt)
                check_dims(LDLt, 2, k-e, k-e)
                print_v("eig_val", eig_val)
                check_dims(eig_val, 1, k-e) # eig_val es un vector del cuál solo extraemos el valor mínimo por lo que no debería haber problema se quede en 1 dimensión
                print_v("lambda_hat_min", lambda_hat_min)
                check_dims(lambda_hat_min, 0)

                if lambda_hat_min > 0:
                    gamma = max(0.5*lambda_hat_min, 1e-6)
                else:
                    gamma = min(1.5*lambda_hat_min, -1e-6)
                print_v("gamma", gamma)
                check_dims(gamma, 0)

                Minv = (LDLt - gamma * SS)  # Minv = (L+D+Lt-St@B@S)
                print_v("Minv", Minv)
                check_dims(Minv, 2, k-e, k-e)
                #print("Minv", Minv, Minv.ndim, Minv.shape, RANK(Minv))
                Psi = Y - gamma * S         # Psi = Y-B@S
                #print("Psi", Psi, Psi.ndim, Psi.shape, RANK(Psi))
                if Psi.ndim == 1:
                    Psi = np.reshape(Psi, (Psi.shape[0], 1))
                    #print("Psi", Psi, Psi.ndim, Psi.shape, RANK(Psi))
                # Psi debería ser de tamaño num de variables x num de iteraciones
                print_v("Psi", Psi)
                check_dims(Psi, 2, n, k-e)

                # Se verifica que las matrices sean de rango completo
                RANK_Psi = RANK(Psi)
                RANK_Minv = RANK(Minv)
                if ((Psi.ndim <= 1 and RANK_Psi == 1) or (Psi.ndim > 1 and Psi.shape[1] == RANK(Psi))) and \
                   ((Minv.ndim <= 1 and RANK_Minv == 1) or (Minv.ndim > 1 and Minv.shape[1] == RANK(Minv))):
                    break
                else:
                    # Eliminamos vectores de curvatura para recalcular las matrices
                    print("Eliminando vectores de curvatura")
                    # Verificamos previo a la eliminación
                    print_v("S", S)
                    check_dims(S, 2, n, k-e)
                    print_v("Y", Y)
                    check_dims(Y, 2, n, k-e)
                    e = e + 1
                    S = S[:, 1:]
                    Y = Y[:, 1:]
                    # Verificamos posterior a la eliminación
                    print_v("S", S)
                    check_dims(S, 2, n, k - e)
                    print_v("Y", Y)
                    check_dims(Y, 2, n, k - e)

    return np.array(Theta)

def TRsubproblem_solver_OBS(delta, gamma, g, Psi, Minv, n, k, e):
    obs_eps = 1e-10
    print("Subproblem solver!")
    # Descomposición
    #print("Psi", Psi, Psi.shape, Psi.ndim)
    if Psi.ndim == 1:
        #print("Reshaping...")
        Psi = np.reshape(Psi, (Psi.shape[0],1))
    #print("Psi", Psi, Psi.shape, Psi.ndim)
    Q, R = QR(Psi, mode="economic")
    print_v("Q", Q)
    check_dims(Q, 2, n, k-e-1) # Restamos 1 porque el solver se ejecuta a partir de la 2a iteración
    print_v("R", R)
    check_dims(R, 2, k-e-1, k-e-1)
    #print("Q", Q, Q.shape, Q.ndim)
    #print("R", R, R.shape, R.ndim)
    #print("Minv", Minv, Minv.shape, Minv.ndim)
    if Minv.ndim == 0:
        RMR = R * (R.T / Minv)
    else:
        RMR = R.dot(SOLVE(Minv, R.T))
    #print("RMR", RMR, RMR.shape, RMR.ndim)
    RMR = (RMR + RMR.T)/2
    print_v("RMR", RMR)
    check_dims(RMR, 2, k-e-1, k-e-1)
    #print("RMR", RMR, RMR.shape, RMR.ndim)
    # Eingenvalores, Eigenvectores
    W, VR = EIG(RMR, right=True)
    print_v("W", W)
    check_dims(W, 1, k-e-1)
    print_v("VR", VR)
    check_dims(VR, 2, k-e-1, k-e-1)
    #print("W", W, W.shape, W.ndim)
    #print("VR", VR, VR.shape, VR.ndim)
    Wd = W
    #print("Wd", Wd, Wd.shape, Wd.ndim)
    lambda_hat = np.sort(Wd)
    print_v("lambda_hat", lambda_hat)
    check_dims(lambda_hat, 1, k-e-1)
    #print("lambda_hat", lambda_hat, lambda_hat.shape, lambda_hat.ndim)
    # TODO revisar cómo obtener los índices (está dando resultados vectores)
    idxs = np.argsort(Wd)
    print_v("idxs", idxs)
    check_dims(idxs, 1, k-e-1)
    #print("idxs", idxs, idxs.shape, idxs.ndim)
    # TODO revisar por qué U resulta de 3 dimensiones
    U = VR[:,idxs]
    print_v("U", U)
    check_dims(U, 2, k-e-1, k-e-1)
    #print("U", U, U.shape, U.ndim)
    lambda1 = lambda_hat + gamma
    print_v("lambda1", lambda1)
    check_dims(lambda1, 1, k-e-1)
    #print("lambda1", lambda1, lambda1.shape, lambda1.ndim)
    lambdap = np.append(lambda1, gamma)
    print_v("lambdap", lambdap)
    check_dims(lambdap, 1, k-e) # No le restamos uno ya que se le añadió un elemento
    #print("lambdap", lambdap, lambdap.shape, lambdap.ndim)
    # Mínimo eigenvalor
    lambdap = lambdap * (np.abs(lambdap) > obs_eps)
    print_v("lambdap", lambdap)
    check_dims(lambdap, 1, k-e) # No le restamos uno ya que se le añadió un elemento
    lambda_min = min(lambdap[0], gamma)
    print_v("lambda_min", lambda_min)
    check_dims(lambda_min, 0)
    #
    #print("Q =", Q, Q.shape, Q.ndim)
    #print("U =", U, U.shape, U.ndim)
    P_ll = Q.dot(U)
    print_v("P_ll", P_ll)
    check_dims(P_ll, 2, n, k-e-1)
    #print("P_ll =", P_ll, P_ll.shape, P_ll.ndim)
    g_ll = P_ll.T.dot(g)
    print_v("g_ll", g_ll)
    check_dims(g_ll, 2, k-e-1, 1)
    #print("g_ll =", g_ll, g_ll.shape, g_ll.ndim)
    gTg = g.T.dot(g)
    print_v("gTg", gTg)
    check_dims(gTg, 2, 1, 1) # Debería ser escalar
    #print("gTg =", gTg, gTg.shape, gTg.ndim)
    g_llTg_ll = g_ll.T.dot(g_ll)
    print_v("g_llTg_ll", g_llTg_ll)
    check_dims(g_llTg_ll, 2, 1, 1) # Debería ser escalar
    #print("g_llTg_ll =", g_llTg_ll, g_llTg_ll.shape, g_llTg_ll.ndim)
    llg_perbll = np.sqrt(np.abs(gTg - g_llTg_ll))
    print_v("llg_perbll", llg_perbll)
    check_dims(llg_perbll, 2, 1, 1) # Debería ser un escalar
    #print("llg_perbll =", llg_perbll, llg_perbll.shape, llg_perbll.ndim)
    if llg_perbll**2 < obs_eps:
        llg_perbll = 0
    a = np.append(g_ll, llg_perbll)
    print_v("a", a)
    check_dims(a, 1, k-e)
    # TODO aquí vamos
    # Case 1
    if (lambda_min > 0) and (phi(0, delta, a, lambdap) >= 0):
        sigma_star = 0
        tau_star = gamma + sigma_star
        p_star = equation_p1(tau_star, g, Psi, Minv)
    # Case 2
    elif (lambda_min <= 0) and (phi(-lambda_min, delta, a, lambdap) >= 0):
        sigma_star = -lambda_min
        p_star = equation_p2(sigma_star, gamma, g, a, lambdap, P_ll, g_ll)
    # Case 3
        if lambda_min < 0:
            p_hat = p_star
            p_star = equation_p3(lambda_min, delta, p_hat, lambdap, P_ll)
    else:
        if lambda_min > 0:
            sigma_star = newton_method(0, delta, a, lambdap)
        else:
            sigma_hat = max(np.abs(a)/delta - lambdap)
            if sigma_hat > -lambda_min:
                sigma_star = newton_method(sigma_hat, delta, a, lambdap)
            else:
                sigma_star = newton_method(-lambda_min, delta, a, lambdap)
        tau_star = sigma_star + gamma
        p_star = equation_p1(tau_star, g, Psi, Minv)
    return p_star

# OBS phi function definition
# Erway, et. al., equation 15, p. 7,
def phi(sigma, delta, a, lambdap):
    obs_eps = 1e-10
    t = lambdap + sigma
    # Zero in a fraction
    if (np.sum(np.abs(a) < obs_eps) > 0) or (np.sum(np.abs(t) < obs_eps) > 0):
        llpll2 = 0
        for i in range(max(a.shape)):
            if (np.abs(a[i]) > obs_eps) and (np.abs(t[i]) < obs_eps):
                return -1/delta
            elif (np.abs(a[i]) > obs_eps) and (np.abs(t[i]) > obs_eps):
                llpll2 = llpll2 + (a[i]/t[i])**2
        return 1/np.sqrt(llpll2) - 1/delta
    # No zero
    llpll = NORM(a/t)
    return 1/llpll - 1/delta

# Equations for linear system solvers
def equation_p1(tau, g, Psi, Minv):
    Z = tau * Minv + Psi.T.dot(Psi)
    f = Psi.T.dot(g)
    return -(g - Psi.dot(SOLVE(Z, f))) / tau
def equation_p2(sigma, gamma, g, a, lambdap, P_ll, g_ll):
    eq_eps = 1e-10
    t = lambdap + sigma
    idx = np.argwhere(np.abs(t) > eq_eps)
    c = max(t.shape)
    v = np.zeros((c,1))
    v[idx] = a[idx]/(lambdap[idx]+sigma)
    if np.abs(gamma + sigma) < eq_eps:
        p = -P_ll * v[0:c-1]
    else:
        p = -P_ll * v[1:c-1] - (g - P_ll.dot(g_ll))/(gamma+sigma)
    return p
def equation_p3(lambda_min, delta, p_hat, lambdap, P_ll):
    eq_eps = 1e-10
    alpha = np.sqrt(delta**2 - p_hat.T.dot(p_hat))
    # Case 1: lambda[0]
    if (np.abs(lambda_min - lambdap[0]) < eq_eps):
        u_min = P_ll[:,0]/NORM(P_ll[:,0])
        z_star = alpha * u_min
    # Case 2: gamma
    else:
        n,k = P_ll.shape
        e = np.zeros((n,1))
        found = 0
        for i in range(k):
            e[i] = 1
            u_min = e - P_ll.dot(P_ll[i,:].T)
            if (NORM(u_min) > eq_eps):
                found = 1
                break
            e[i] = 0
        if found == 0:
            e[i+1] = 1
            u_min = e - P_ll.dot(P_ll[i+1,:].T)
        u_min = u_min/NORM(u_min)
        z_star = alpha * u_min
    p_star = p_hat + z_star
    return p_star
def newton_method(sigma_0, delta, a, lambdap):
    newton_eps = 1e-10
    newton_maxit = 100
    k = 0
    sigma = sigma_0
    phi_f, phi_prim = phi_phiprim(sigma, delta, a, lambdap)
    while (np.abs(phi_f) > newton_eps) and (k < newton_maxit):
        sigma = sigma - phi_f/phi_prim
        phi_f, phi_prim = phi_phiprim(sigma, delta, a, lambdap)
        k = k + 1
    return sigma
def phi_phiprim(sigma, delta, a, lambdap):
    obs_eps = 1e-10
    t = lambdap + sigma
    # Zero fraction
    if (np.sum(np.abs(a) < obs_eps) > 0) or (np.sum(np.abs(t) < obs_eps) > 0):
        llpll2 = 0
        llpll_prim = 0
        for i in range(max(a.shape)):
            if (np.abs(a[i]) > obs_eps) and (np.abs(t[i]) < obs_eps):
                phi_f = -1/delta
                phi_prim = 1/obs_eps
                return phi_f, phi_prim
            elif (np.abs(a[i]) > obs_eps) and (np.abs(t[i]) > obs_eps):
                llpll2 = llpll2 + (a[i]/t[i])**2
                llpll_prim = llpll_prim + ((a[i]**2)/(t[i]**3))
        llpll = np.sqrt(llpll2)
        phi_f = 1/llpll - 1/delta
        phi_prim = llpll_prim/(llpll**3)
        return phi_f, phi_prim
    # No Zero fraction
    llpll = NORM(a/t)
    phi_f = 1/llpll - 1/delta
    phi_prim = np.sum((a**2)/(t**3))/(llpll**3)
    return phi_f, phi_prim

# Algorithm 1.- Stochastic SR1 Trust-region scheme (MB-LSR1)
# Griffin, et. al., p. 5
def MB_LSR1(theta=[], fun=None, grad=None, gd_params={}, f_params={}):
    '''
    Stochastic SR1 Trust-region scheme (MB-LSR1)
    Griffin, et. al.
    A minibatch stochastic Quasi-Newton method adapted for nonconvex deep learning problems
    Algorithm 1, p. 5

    Parámetros
    -----------
    theta     :   condicion inicial
    fun       :   función de costo
    grad      :   gradiente de la función de costo
    gd_params :   lista de parametros para el algoritmo de descenso,
                    nIter    = gd_params['nIter'] número de iteraciones
                    X,y:                Observations and labels of the data to classify
                    seed:               Random seeder
                    numIter:            Max. num. of iterations
                    mmr:                Memory size
                    r_mmr:              Restart memory size r_mmr <= m
                    n:                  Initial batch size
                    tau:                Restart tolerance
                    K:                  Progress check frequency
                    gamma_1, gamma2:    Progress threshold parameters
                    zeta:               Progressive radius parameter [0, 1]
                    mu:                 Momentum
                    alpha_s:            Learning rate [0, 1]
                    radius:             Radius of the sampling region
                    eps:                Epsilon criterion to sample S,Y pairs
                    eta:                Trust region increase/reduce criterion
                    delta_init:         Initial radius of trust region
                    epsTR:              Epsilon criterion for trust region subproblem
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
    # Line 1.- Input
    N       = gd_params['N']        # Num iterations
    mmr     = gd_params['mmr']      # Memory
    mmr_h   = gd_params['mmr_h']    # Restart memory
    n       = gd_params['n']        # Initial batch-size
    delta_0 = gd_params['delta_0']  # Initial radius (delta_0)
    # Line 2.- Parameters
    tau     = gd_params['tau']      # Restart tolerance
    K       = gd_params['K']        # Progress check frequency
    gamma_1 = gd_params['gamma_1']  # Progress threshold parameter 1
    gamma_2 = gd_params['gamma_2']  # Progress threshold parameter 2
    nh_k    = gd_params['nh_k']     # Progress check batch-size
    zeta    = gd_params['zeta']     # Progressive radius parameter [0, 1]
    eta     = gd_params['eta']      # Step parameter
    mu      = gd_params['mu']       # Momentum
    alpha_s = gd_params['alpha_s']  # Learning-rate [0, 1]
    # Parámetros adicionales
    eps     = gd_params['eps']      # Termination Check Tolerance

    # Line 3.- Variables
    v = 0.0
    T = 0
    k = 0
    rho_h = 0
    s = y = v = 0
    S = Y = []
    Theta = []
    # Line 4
    for t in range(N):
        # Line 5
        # p = np.zeros(theta.shape)
        s, y, v, rho, R = getStep(
            # Parámetros del algoritmo
            s, v, delta, eta, mu, alpha_s,
            # Parámetros de la función objetivo y gradiente
            fun, grad, theta, f_params
        )
        # Line 6
        thetaNew = theta + s

        # Line 7
        if (exitCheck(grad, theta, f_params, eps) == 1): break
        # Line 8
        delta, rho_k, T = getRadius(delta, s, rho, zeta, rho_k, T)
        # Line 9
        S, Y = updateSY(S, Y, s, y, rho, tau, mmr_h)
        # Line 10
        n, fK, zeta = getBatch(n, N, fK, k, K, R, gamma_1, gamma_2, zeta, fun, grad, theta, f_params)
        # Update variables
        theta = thetaNew
        Theta.append(theta)
        k = k + 1
    return np.array(Theta)

# Algorithm 4 Search direction computation algorithm
# Griffin, et. al., p. 6
def getStep(
        # Parámetros del algoritmo
        s, v, delta, eta, mu, alpha_s,
        # Parámetros adicionales del algoritmo
        Eps, S, Y,
        # Parámetros de la función objetivo y gradiente
        fun, grad, theta, f_params):
    '''
        Parameters:
            s: vector
            v: vector
            delta: radius
            eta: threshold
            mu: momentum
            alpha_s: learning rate
            eps: epsilon
            grad_k: batch gradient
            S,Y: curvature pairs
    '''
    # Line 2.- Compute batch gradient gk = \nabla f(wk)
    gk = grad(theta, f_params)
    # Line 3.- Compute p* by applying the algorithm proposed in [8] using latest S, Y pairs
    # TODO Implement calculation of p*
    ps = calculate_pstar(S,Y)
    # Line 4
    v = mu * v - eta * alpha_s * gk + (1 - eta)*s
    # Line 5
    v = min(1, delta/np.linalg.norm(v))*v
    # Line 6
    p = (1 - eta)*ps + mu * v
    # Line 7
    p = min(1, delta/np.linalg.norm(p))*p
    # Line 8
    if p.T.dot(gk) > 0:
        # Line 9
        p = -p
    gkTp = gk.T.dot(p)
    # Line 11
    if min(np.abs(gkTp),np.abs(gkTp)/(np.linalg.norm(p)**2)) < Eps*(np.linalg.norm(gk)**2):
        # Line 12.- restart LSR1 with initial matrix being I
        # Line 13.- return and restart getStep
        return
    # Line 15.- Get step-size alpha from line-search on current batch
    alpha = 1
    c1 = 1e-4
    rho_ls = 0.5
    # Line search on Armijo condition f(wk + alpha*p) <= f(wk) + c1*alpha*( grad(wk).T @ p)
    while fun(theta + alpha * p, f_params) > fun(theta, f_params) + c1 * alpha * (gk.T.dot(p)):
        alpha = rho_ls * alpha
    # Line 16
    s = alpha * p
    # Line 17
    # Compute actual reduction
    R = fun(theta + ps, f_params) - fun(theta, f_params)
    # Compute predicted reduction
    Lp = np.zeros((Y.shape[1], Y.shape[1]))
    for ii in range(Y.shape[1]):
        for jj in range(0, ii):
            Lp[ii, jj] = S[:, ii].dot(Y[:, jj])
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + Lp + Lp.T)
    Minvp = np.linalg.inv(Mp)
    tmpp1 = np.matmul(Y.T, ps)
    tmpp2 = np.matmul(Minvp, tmpp1)
    Bk_ps = np.matmul(Y, tmpp2)
    Q = -(gk.T.dot(ps) + 0.5 * ps.T.dot(Bk_ps))  # Compute predicted reduction
    # Line 18
    rho = R / Q
    # Line 19
    y = grad(theta + s, f_params) - gk
    return s, y, v, rho, R

# Algorithm 2 Termination Check
# Griffin, et. al., p. 5
def exitCheck(grad, theta, f_params, eps):
    # Line 3
    # TODO Implement batch gradient
    if np.linalg.norm(grad(theta, f_params)) <= eps:
        # Line 4, 5.- Evaluate new gk over whole data set
        if np.linalg.norm(grad(theta, f_params)) <= eps:
            # Line 6: Stop
            return 1
    # Line 9: Not stop
    return 0

# Algorithm 5 Example update radius function
# Griffin, et. al. p. 7
def getRadius(delta, s, rho, zeta, rho_h, T):
    # Line 2.- update non-monotone ratio threshold rho_h
    # Line 3
    rho_h = zeta * T * rho_h + rho
    # Line 4
    T = zeta * T + 1
    # Line 5
    rho_h = rho_h / T
    ns = np.linalg.norm(s)
    # Line 6
    if rho_h < 0.1:
        # Line 7
        delta = min(delta, ns)
    else:
        if rho_h >= 0.5 and ns >= delta:
            delta = 2 * delta
    return delta, rho_h, T

# Algorithm 3 Pair selection algorithm
# Griffin, et. al. p. 6
def updateSY(S, Y, s, y, rho, tau, k, mmr, mmr_h):
    # Line 2
    if rho < tau and mmr_h >= 0:
        if mmr_h == mmr:
            # Line 4.- Warm start according to [1]
            # Line 5: Generate m pairs and store in S and Y
            # TODO Implement generation of m pairs
            pass
        # Line 6
        else:
            # Line 6
            if mmr_h == 0:
                # Line 7
                S = []
                Y = []
                # Restart
    else:
        # Classic S, Y update
        # Line 11
        S.append(s)
        Y.append(y)
        # Line 12
        if k > mmr:
            S.pop(0)
            Y.pop(0)
    # Line 14
    return S,Y

# Algorithm 6 Example batch-size correction function
# Griffin, et. al. p. 7
def getBatch(nb, N, fK, k, K, R, gamma_1, gamma_2, zeta, fun, grad, theta, f_params):
    # Line 2
    if K % k == 0:
        # Line 3.- Randomly draw new progress check sample of size nh = min(1.1*nb, 5000)
        # Line 4.- Define f using the new sample
        # Line 5
        sumRj = 0
        for j in range(K-k,k):
            sumRj = sumRj + R[j]
        if grad(theta, f_params) - fK >= -gamma_1 + gamma_2 * sumRj:
            # Line 6.- Progress worse than simple line-search
            # Line 7
            nb = min(2*nb, N)
            # Line 8
            if nb == N:
                # Line 9
                zeta = 0
        # Line 12.- Update new target bound fK
        # Line 13
        fK = fun(theta, f_params)
    # Line 15.- Randomly draw batch Ik+1 with size nb defining f^
    return nb, fK, zeta



# Algorithm 2: Orthonormal Basis SR1 Method
def Orthonormal_Basis(S,Y, gamma):
    Psi = Yk - B0 @ Sk
    # 1.- Compute the Cholesky factor R of Psi' * Psi
    R = np.linalg.cholesky(Psi.T @ Psi)
    Rinv = np.linalg.inv(R)
    # Requerimos construir la matriz M para el paso siguiente
    tmpp = np.sum((S * Y), axis=0)
    Dp = np.diag(tmpp)
    Mp = (Dp + R + R.T)
    Minvp = np.linalg.inv(Mp)
    # 2.- Compute the spectral decomposition R*M*R' = U*Â*U' (solo usamos los eingenvalores)
    A, U = np.linalg.eig(R @ (Minvp @ R.T))
    A1 = A + gamma * np.eye(len(A))
    lambda_1 = np.min(A1)
    lambda_min = min(lambda_1, gamma)
    PsiRinvU = Psi @ (Rinv @ U)
    gb = PsiRinvU.T @ gk
    if lambda_min > 0 and psi(0) >= 0:
        sigma_s = 0
        Bkinv = np.linalg.inv(Bk)
        ps = - Bkinv @ gk
    else:
        pass
