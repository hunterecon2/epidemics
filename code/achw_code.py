import numpy as np
import matplotlib.pyplot as plt

def achewwmodel(s0=[.53, .25, .21], i0=[0, .01, 0], rho=[[.2, .2, .2], [.2, .2, .2], [.2, .2, .2]], 
                theta=[.5, .5, .5], taper=[0.0,0.0,0.0], L=[0, 0, 0], delay=0, maxlock=20, sigma=40, lam=700, 
                lf=[.8, .72, .15], its=1000, epsilon=[0, 0, 0],
                gamma=[0.054, 0.054, 0.054], tau=[0, 0, 0], phi=[0, 0, 0], kappa = [0, 0, 0], wf=[.1, .2, .05],
                dd=[0.0000555, 0.000555, 0.00333], alpha=2,  tol=.001):

    '''This function simulates the ACHEWW model approximately in discrete daily time. 
       One should put in the maximal values of the lockdown parameters theta, and then 
       allow for taper (which functions like a rate of depreciation) after that. Additional 
       policy parameters are static in that we suppose there is a fixed policy about 
       isolation/detection/release. Taper can also be specific to age group, note.

       Description:
       s0      -> fractions of each population that is initially susceptible
       i0      -> fraction of each population age group initially infected
       rho     -> base social interaction rate between age classes. Diagonal: within group.
                  off diagonal: between group. Should be a symmetric matrix.
       theta   -> degree to which there is leakage in a policy
       taper   -> rate at which a lockdown "depreciates" by the day
       L       -> Lockdown parameters for each age grade. can be partial, or total if L=1,1,1,
       delay   -> Time epidemic runs before lockdown kicks in
       maxlock -> Length of time in days lockdown persists at its maximum value
       sigma   -> parameter translating hospitalization into deaths
       lam     -> parameter translating hospitalization into deaths
       lf      -> Labor force participation rate of each age grade
       its     -> maximum model iterations in days
       epsilon -> Persistence of employment parameter
       gamma   -> daily rate at which infecteds recover
       tau     -> rate at which hospitalized cases can be detected and isolated
       phi     -> rate at which non-hospitalized cases can be detected and isolated
       kappa   -> rate at which recovered can be detected and released from lockdown
       wf      -> Fraction of age grade that can work from home
       dd      -> death rate by age grade
       alpha   -> matching parameter. CRS if alpha=1, Quadratic if alpha=2
       tol     -> Tolerance for stopping the run as the epidemic has run its course.
       '''

    iota = [sigma*deathrate for deathrate in dd]
    
    eta = [1 - iota[0]*phi[0] - (1 - iota[0])*tau[0] ,
           1 - iota[1]*phi[1] - (1 - iota[1])*tau[1] ,
           1 - iota[2]*phi[2] - (1 - iota[2])*tau[2] ]
    
    beta = [[r*eta[0] for r in rho[0]],
            [r*eta[1] for r in rho[1]],
            [r*eta[2] for r in rho[2]] ]
    
    S1 = []
    S2 = []
    S3 = []

    I1 = []
    I2 = []
    I3 = []
    
    R1 = []
    R2 = []
    R3 = []
    
    H1 = []
    H2 = []
    H3 = []
    
    D1 = []
    D2 = []
    D3 = []
    
    L1 = []
    L2 = []
    L3 = []
    
    U1 = []
    U2 = []
    U3 = []
    UT = []
    
    s1, s2, s3 = s0[0], s0[1], s0[2]
    i1, i2, i3 = i0[0], i0[1], i0[2]
    r1, r2, r3 = 0, 0, 0
    d1, d2, d3 = 0, 0, 0
    u1, u2, u3, ut = 0, 0, 0, 0
    
    t=0
    
    i = i1 + i2 + i3
    
    lockcount = 0
    tapecount = 0
    
    while t<its and i> tol: 
        
        if t <  delay:
            la = [0, 0, 0]
        else:
            la = [1, 1, 1]
            lockcount += 1
            
        if lockcount <= maxlock:
            l = [lla*LLa for lla, LLa in zip (la, L)]
        else:
            l = [(lla*LLa)*(1-tape)**tapecount for lla, LLa, tape in zip (la, L, taper)]
            tapecount += 1
        
        L1.append(l[0])
        L2.append(l[1])
        L3.append(l[2])
        
        m1 = ( beta[0][0]*((s1 + eta[0]*i1 + (1-kappa[0])*r1)*(1 - l[0]*theta[0]) + kappa[0]*r1) + 
               beta[0][1]*((s2 + eta[1]*i2 + (1-kappa[1])*r2)*(1 - l[1]*theta[1]) + kappa[1]*r2) +
               beta[0][2]*((s3 + eta[2]*i3 + (1-kappa[2])*r3)*(1 - l[2]*theta[2]) + kappa[2]*r3)   )**(alpha - 2)
        m2 = ( beta[1][0]*((s1 + eta[0]*i1 + (1-kappa[0])*r1)*(1 - l[0]*theta[0]) + kappa[0]*r1) +
               beta[1][1]*((s2 + eta[1]*i2 + (1-kappa[1])*r2)*(1 - l[1]*theta[1]) + kappa[1]*r2) +
               beta[1][2]*((s3 + eta[2]*i3 + (1-kappa[2])*r3)*(1 - l[2]*theta[2]) + kappa[2]*r3)   )**(alpha - 2)
        m3 = ( beta[2][0]*((s1 + eta[0]*i1 + (1-kappa[0])*r1)*(1 - l[0]*theta[0]) + kappa[0]*r1) +
               beta[2][1]*((s2 + eta[1]*i2 + (1-kappa[1])*r2)*(1 - l[1]*theta[1]) + kappa[1]*r2) +
               beta[2][2]*((s3 + eta[2]*i3 + (1-kappa[2])*r3)*(1 - l[2]*theta[2]) + kappa[2]*r3)   )**(alpha - 2)
        
        num1 = (1 - l[0]*theta[0])*s1*( beta[0][0]*i1*(1 - l[0]*theta[0]) + beta[0][1]*i2*(1 - l[1]*theta[1]) + 
               beta[0][2]*i3*(1 - l[2]*theta[2])    )
        num2 = (1 - l[1]*theta[1])*s2*( beta[1][0]*i1*(1 - l[0]*theta[0]) + beta[1][1]*i2*(1 - l[1]*theta[1]) + 
               beta[1][2]*i3*(1 - l[2]*theta[2])    )
        num3 = (1 - l[2]*theta[2])*s3*( beta[2][0]*i1*(1 - l[0]*theta[0]) + beta[2][1]*i2*(1 - l[1]*theta[1]) + 
               beta[2][2]*i3*(1 - l[2]*theta[2])    )
        
        # Before next-period values let's compute statics like hospitalizations and all that
        # Current hospitalizations
        
        h1 = iota[0]*i1
        h2 = iota[1]*i2
        h3 = iota[2]*i3
        
        h = h1 + h2 + h3

        #### Unemployment variables - these are the "natural" values which we then subject to a lag.
        
        u1n = (1 - wf[0])*l[0]*(s1 + (1 - iota[0]*phi[0] - (1-iota[0])*tau[0])*i1 + (1-kappa[0])*r1) 
        u2n = (1 - wf[1])*l[1]*(s2 + (1 - iota[1]*phi[1] - (1-iota[1])*tau[1])*i2 + (1-kappa[1])*r2) 
        u3n = (1 - wf[2])*l[2]*(s3 + (1 - iota[2]*phi[2] - (1-iota[2])*tau[2])*i3 + (1-kappa[2])*r3) 
        
        utn = ( (lf[0]*u1n*(s1 + i1 + r1) + lf[1]*u2n*(s2 + i2 + r2) + lf[2]*u3n*(s3 + i3 + r3) ) /
               (lf[0]*(s1 + i1 + r1) + lf[1]*(s2 + i2 + r2) + lf[2]*(s3 + i3 + r3)) )
        
        psi  = 3*[None]
        psi[0] = dd[0]*(1 + lam*h)
        psi[1] = dd[1]*(1 + lam*h)
        psi[2] = dd[2]*(1 + lam*h)
        
        dr = [ (gam-ps) / (1-ps) for (gam, ps) in zip (gamma, psi)]

        ##### Update to all the variables
        
        i1p = i1 - i1*(1 - iota[0])*gamma[0] - h1*(psi[0] + (1-psi[0])*dr[0]) + m1*num1
        i2p = i2 - i2*(1 - iota[1])*gamma[1] - h2*(psi[1] + (1-psi[1])*dr[1]) + m2*num2
        i3p = i3 - i3*(1 - iota[2])*gamma[2] - h3*(psi[2] + (1-psi[2])*dr[2]) + m3*num3
        
        s1p = s1 - m1*num1
        s2p = s2 - m2*num2
        s3p = s3 - m3*num3

        r1p = r1 + (1 - iota[0])*gamma[0]*i1 + h1*(1-psi[0])*dr[0]
        r2p = r2 + (1 - iota[1])*gamma[1]*i2 + h2*(1-psi[1])*dr[1]
        r3p = r3 + (1 - iota[2])*gamma[2]*i3 + h3*(1-psi[2])*dr[2]

        d1p = d1 + h1*psi[0]
        d2p = d2 + h2*psi[1]
        d3p = d3 + h3*psi[2] 
        
        u1p = (1 - epsilon[0])*u1n + epsilon[0]*u1
        u2p = (1 - epsilon[1])*u2n + epsilon[1]*u2
        u3p = (1 - epsilon[2])*u3n + epsilon[2]*u3 
        
        eps = np.sum(epsilon)/3
        
        utp = (1 - eps)*utn + eps*ut
        
        S1.append(s1p)
        S2.append(s2p)
        S3.append(s3p)

        I1.append(i1p)
        I2.append(i2p)
        I3.append(i3p)

        R1.append(r1p)
        R2.append(r2p)
        R3.append(r3p)

        D1.append(d1p)
        D2.append(d2p)
        D3.append(d3p)
        
        H1.append(h1)
        H2.append(h2)
        H3.append(h3)
        
        U1.append(u1p)
        U2.append(u2p)
        U3.append(u3p)
        UT.append(utp)
        
        s1 = s1p
        s2 = s2p
        s3 = s3p

        i1 = i1p
        i2 = i2p
        i3 = i3p

        r1 = r1p
        r2 = r2p
        r3 = r3p

        d1 = d1p
        d2 = d2p
        d3 = d3p
        
        i = i1p + i2p + i3p
        t += 1
    
    agg_u   = np.sum(UT)
    agg_d   = np.sum(D1[-1] + D2[-1] + D3[-1])
    agg_r   = np.sum(R1[-1] + R2[-1] + R3[-1])
    days    = t
    maxinf  = np.max([inf1 + inf2 + inf3 for inf1, inf2, inf3 in zip(I1, I2, I3)])
    maxhos  = np.max([hos1 + hos2 + hos3 for hos1, hos2, hos3 in zip(H1, H2, H3)])
    maxinfd = np.argmax([inf1 + inf2 + inf3 for inf1, inf2, inf3 in zip(I1, I2, I3)])
    maxhosd = np.argmax([hos1 + hos2 + hos3 for hos1, hos2, hos3 in zip(H1, H2, H3)])
    
    return({'S':{'s1':S1, 's2':S2, 's3':S3}, 'I':{'i1':I1, 'i2':I2, 'i3':I3},
            'R':{'r1':R1, 'r2':R2, 'r3':R3}, 'D':{'d1':D1, 'd2':D2, 'd3':D3}, 
            'H':{'h1':H1, 'h2':H2, 'h3':H3}, 'L':{'l1':L1, 'l2':L2, 'l3':L3},
            'U':{'u1':U1, 'u2':U2, 'u3':U3, 'ut':UT},
            'Aggs':{'agg_u':agg_u,'agg_d':agg_d, 'agg_r': agg_r,
                    'days':days, 'maxinf':maxinf,
                    'maxhos': maxhos, 'maxinfd':maxinfd,
                    'maxhosd':maxhosd}})  

def compare_models_detailed(res, resp):
    ''' Takes in two sets of results and plots them for purposes of comparison
        in a set of six side-by-side plots.'''
    
    fig = plt.figure(figsize=(15, 12))
    axs = fig.add_subplot(3, 2, 1)
    adj = fig.subplots_adjust(bottom=-.4)
    xr  = np.arange(len(res['S']['s1']))
    ls1 = axs.plot(xr, res['S']['s1'])
    ls2 = axs.plot(xr, res['S']['s2'])
    ls3 = axs.plot(xr, res['S']['s3'], linewidth=4)

    stotal = [s1 + s2 + s3 for s1, s2, s3 in zip(res['S']['s1'] , res['S']['s2'], res['S']['s3'])]

    lst = axs.plot(xr, stotal)

    xrp  = np.arange(len(resp['S']['s1']))
    ls1p = axs.plot(xrp, resp['S']['s1'], linestyle='--')
    ls2p = axs.plot(xrp, resp['S']['s2'], linestyle='--')
    ls3p = axs.plot(xrp, resp['S']['s3'], linestyle='--', linewidth=4)

    stotalp = [s1 + s2 + s3 for s1, s2, s3 in zip(resp['S']['s1'] , resp['S']['s2'], resp['S']['s3'])]

    lstp = axs.plot(xrp, stotalp, linestyle='--')

    sla = axs.legend(['Young', 'Middle', 'Old', 'Total', 'Young', 'Middle', 'Old', 'Total'])
    stit = axs.title.set_text('Susceptible')

    axi = fig.add_subplot(3, 2, 2)

    xr  = np.arange(len(res['I']['i1']))
    li1 = axi.plot(xr, res['I']['i1'])
    li2 = axi.plot(xr, res['I']['i2'])
    li3 = axi.plot(xr, res['I']['i3'],linewidth=4 )

    itotal = [i1 + i2 + i3 for i1, i2, i3 in zip(res['I']['i1'] , res['I']['i2'], res['I']['i3'])]

    lst = axi.plot(xr, itotal)

    xrp  = np.arange(len(resp['I']['i1']))
    li1p = axi.plot(xrp, resp['I']['i1'], linestyle='--')
    li2p = axi.plot(xrp, resp['I']['i2'], linestyle='--')
    li3p = axi.plot(xrp, resp['I']['i3'], linestyle='--', linewidth=4)

    itotalp = [i1 + i2 + i3 for i1, i2, i3 in zip(resp['I']['i1'] , resp['I']['i2'], resp['I']['i3'])]

    lstp = axi.plot(xrp, itotalp, linestyle='--')

    ila = axi.legend(['Young', 'Middle', 'Old', 'Total', 'Young', 'Middle', 'Old', 'Total'])
    itit = axi.title.set_text('Infected')

    axr = fig.add_subplot(3, 2, 3)

    xr  = np.arange(len(res['R']['r1']))
    lr1 = axr.plot(xr, res['R']['r1'])
    lr2 = axr.plot(xr, res['R']['r2'])
    lr3 = axr.plot(xr, res['R']['r3'], linewidth=4)

    rtotal = [r1 + r2 + r3 for r1, r2, r3 in zip(res['R']['r1'] , res['R']['r2'], res['R']['r3'])]

    lst = axr.plot(xr, rtotal)

    xrp  = np.arange(len(resp['R']['r1']))
    lr1p = axr.plot(xrp, resp['R']['r1'], linestyle='--')
    lr2p = axr.plot(xrp, resp['R']['r2'], linestyle='--')
    lr3p = axr.plot(xrp, resp['R']['r3'], linestyle='--', linewidth=4)

    rtotalp = [r1 + r2 + r3 for r1, r2, r3 in zip(resp['R']['r1'] , resp['R']['r2'], resp['R']['r3'])]

    lst = axr.plot(xrp, rtotalp, linestyle='--')

    rla = axr.legend(['Young', 'Middle', 'Old', 'Total', 'Young', 'Middle', 'Old', 'Total'])
    rtit = axr.title.set_text('Recovered')

    axd = fig.add_subplot(3, 2, 4)

    xr  = np.arange(len(res['D']['d1']))
    ld1 = axd.plot(xr, res['D']['d1'])
    ld2 = axd.plot(xr, res['D']['d2'])
    ld3 = axd.plot(xr, res['D']['d3'], linewidth=4)
 
    dtotal = [d1 + d2 + d3 for d1, d2, d3 in zip(res['D']['d1'] , res['D']['d2'], res['D']['d3'])]

    lst = axd.plot(xr, dtotal)

    xrp  = np.arange(len(resp['D']['d1']))
    ld1p = axd.plot(xrp, resp['D']['d1'], linestyle='--')
    ld2p = axd.plot(xrp, resp['D']['d2'], linestyle='--')
    ld3p = axd.plot(xrp, resp['D']['d3'], linestyle='--',linewidth=4 )

    dtotalp = [d1 + d2 + d3 for d1, d2, d3 in zip(resp['D']['d1'] , resp['D']['d2'], resp['D']['d3'])]

    lst = axd.plot(xrp, dtotalp, linestyle='--')

    rla = axd.legend(['Young', 'Middle', 'Old', 'Total', 'Young', 'Middle', 'Old', 'Total'])
    rtit = axd.title.set_text('Deaths')

    axh = fig.add_subplot(3, 2, 5)

    xr  = np.arange(len(res['H']['h1']))
    ld1 = axh.plot(xr, res['H']['h1'])
    ld2 = axh.plot(xr, res['H']['h2'])
    ld3 = axh.plot(xr, res['H']['h3'],linewidth=4 )

    htotal = [h1 + h2 + h3 for h1, h2, h3 in zip(res['H']['h1'] , res['H']['h2'], res['H']['h3'])]

    lst = axh.plot(xr, htotal)

    xrp  = np.arange(len(resp['H']['h1']))
    ld1p = axh.plot(xrp, resp['H']['h1'], linestyle='--')
    ld2p = axh.plot(xrp, resp['H']['h2'], linestyle='--')
    ld3p = axh.plot(xrp, resp['H']['h3'], linestyle='--', linewidth=4 )

    htotalp = [h1 + h2 + h3 for h1, h2, h3 in zip(resp['H']['h1'] , resp['H']['h2'], resp['H']['h3'])]

    lst = axh.plot(xrp, htotalp, linestyle='--')

    rla = axh.legend(['Young', 'Middle', 'Old', 'Total', 'Young', 'Middle', 'Old', 'Total'])

    rtit = axh.title.set_text('Hospitalization')

    axpol = fig.add_subplot(3, 2, 6)

    xr  = np.arange(len(res['L']['l1']))
    ld1 = axpol.plot(xr, res['L']['l1'])
    ld2 = axpol.plot(xr, res['L']['l2'])
    ld3 = axpol.plot(xr, res['L']['l3'], linewidth=4)

    lst = axh.plot(xr, htotal)

    xrp  = np.arange(len(resp['L']['l1']))
    ld1p = axpol.plot(xrp, resp['L']['l1'], linestyle='--')
    ld2p = axpol.plot(xrp, resp['L']['l2'], linestyle='--')
    ld3p = axpol.plot(xrp, resp['L']['l3'], linestyle='--', linewidth=4)

    polla = axpol.legend(['Young', 'Middle', 'Old', 'Total', 'Young', 'Middle', 'Old', 'Total'])

    poltit = axpol.title.set_text('Policy')

def display_aggs(results):
    print('Unemployment days lost:  ', results['Aggs']['agg_u']) 
    print('Total deaths:            ', results['Aggs']['agg_d'])
    print('Total epidemic days:     ', results['Aggs']['days'])
    print('Total recovered:         ', results['Aggs']['agg_r'])
    print('Maximum infection at ', results['Aggs']['maxinfd'], ' with value ',  results['Aggs']['maxinf'])
    print('Maximum hospitalizations at ', results['Aggs']['maxhosd'], ' with value ', results['Aggs']['maxhos'])

