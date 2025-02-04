#mcandrew

import numpyro
numpyro.enable_x64(True)

import sys
import numpy as np
import pandas as pd
import os

import pickle

import dask
from dask import delayed, compute

import matplotlib.pyplot as plt
import seaborn as sns

from epiweeks import Week

from datetime import datetime, timedelta
import jax

from scipy.ndimage import gaussian_filter1d

def model(rng_key = None, location=None, cases=None,hosps=None, X=None, VE=None, peak_data=None,ili_n=None,ili_i=None,N=None):
    import numpyro
    numpyro.enable_x64(True)

    num_chains  = 3
    numpyro.set_host_device_count(num_chains)
    
    import numpyro.util
    import numpyro.distributions as dist

    from   numpyro.util import format_shapes
    from   numpyro.handlers import trace, seed
    
    from   numpyro.infer import MCMC, NUTS, init_to_value, init_to_median, init_to_sample,init_to_uniform
    from numpyro.infer.autoguide import AutoNormal, AutoMultivariateNormal
    from numpyro.optim import Adam
    from numpyro.infer import SVI, Trace_ELBO, init_to_median

    from   numpyro.infer import Predictive

    import numpyro.distributions as dist
    
    import jax
    from   jax import jit
    from   jax.random import PRNGKey
    from   jax.scipy.optimize import minimize
    import jax.numpy as jnp
    from   jax.scipy.special import logit,expit

    from   diffrax import ODETerm, SaveAt, diffeqsolve, Euler, Heun

    from   functools import partial
    jax.clear_caches()

    #--fit dynamical system first to fix group means
  
    # Define the solver and solution saving points
    def compute_incidence(init, R0, gamma,phi,alpha, delta, exposed, trt, beta_cov,X, errs, T, ts):
        args     = jnp.array([x.squeeze() for x in (R0,gamma,phi,alpha, delta, exposed,trt)])
        init     = init.squeeze()

        if beta_cov is not None:
            beta_cov = beta_cov.squeeze()
            
        def R0_structure(t,ts,R0, beta_cov=None, X=None,errs=None):
            log_R0 = jnp.log(R0)
            if X is not None:
                def extract_data(t,ts,xs):
                    func_value = jnp.interp(t,ts,xs)     
                    return func_value
                x = jax.vmap( lambda xs: extract_data(t,ts,xs), in_axes=(1) )( X )

                beta_cov = beta_cov.reshape(-1,1)
                x        = x.reshape(-1,1)

                if errs is not None:
                    err_vales = jnp.interp(t,ts,errs)
                    log_R0    = jnp.log(R0) + np.sum(beta_cov*x) + err_vales
                else:
                    log_R0    = jnp.log(R0) + np.sum(beta_cov*x)
            
            return  jnp.clip( jnp.exp(log_R0), 0,10) #<--un-reasonable to be outisde this range
            
        @jit
        def sir_model(t, y, args, key=jax.random.PRNGKey(1)):
            S,St, E1, E2, I, H, R,   C, CILI, CH = y 
            def deriv(args):
                (R0,gamma,phi,alpha, delta, exposed,trt),ts,beta_cov,X,errs = args

                R0     = R0_structure(t,ts,R0,beta_cov,X,errs)
                beta   = gamma*R0

                dSdt   = -beta * (S * I)
                dStdt  = -trt*beta * (St * I)

                dE1dt  =  beta*I*(S + trt*St)       - 2*exposed * E1
                dE2dt  =  2*exposed * E1           - 2*exposed * E2

                dIdt   =  2*exposed * E2           - gamma * I

                dHdt   =  phi*gamma*I    - delta*H
                dRdt   = (1-phi)*gamma*I + delta*H 

                #--helper states to compute incidence
                dCdt    =  2*exposed*E2 
                dCILIdt =  alpha*gamma*I
                dCHdt   =  phi*gamma*I

                return jnp.array([dSdt, dStdt, dE1dt,dE2dt, dIdt, dHdt, dRdt,  dCdt, dCILIdt, dCHdt])
            return deriv(args)
        
        dx      = 1./7
        solver  = diffeqsolve(ODETerm(sir_model)
                             , solver = Euler()
                             , t0     = -1
                             , t1     =  T
                             , dt0    = dx
                             , y0     = init
                             , args   = (args,jnp.linspace(-1,T,T+1), beta_cov, X,errs )
                             , saveat = SaveAt(ts= ts))

        cincident_infections       = solver.ys[...,-3]
        incident_infections        = jnp.diff(cincident_infections,axis=-1)

        cincident_ILI              = solver.ys[...,-2]
        incident_ILIs              = jnp.diff(cincident_ILI,axis=-1)
        
        cincident_hospitalizations = solver.ys[...,-1]
        incident_hospitalizations  = jnp.diff(cincident_hospitalizations,axis=-1)
        
        return incident_infections, incident_ILIs, incident_hospitalizations

    #--loglikelihood function
    def loglikelihood(self,x,cases=None,hosps=None,ili_n=None,ili_i=None):
        eps = 10**-12

        NSAMPLES = x.shape[0]
        
        T        = self.T
        nseasons = self.nseasons
        N        = self.N
        cases    = self.cases
        
        #--inital conitions
        vacc = jnp.full_like(x[:,0], 0.50).reshape(-1,1)   
        
        S0   = (1-vacc)*x[:,0].reshape(-1,1)
        St0  =     vacc*x[:,0].reshape(-1,1)

        I0   = 1. - (S0+St0)

        Z    = jnp.full_like(I0,0.)
        H0   = Z
        
        initial_conditions = jnp.hstack([S0,St0,Z,Z,I0,H0,Z, I0,Z,Z])

        #--dynamical system paramaters
        dynamics = {}
        dynamics['R0']        = jnp.array(x[:,1]).reshape(-1,1)
        dynamics['gamma']     = (3./7)*jnp.ones( (NSAMPLES,1) )                #--infectious period is something like 7 days or one week #jnp.array(x[:,2]).reshape(-1,1)  
        
        dynamics['phi']       = jnp.array(x[:,2]).reshape(-1,1)
        dynamics['alpha']     = jnp.array(x[:,3]).reshape(-1,1)
        
        dynamics['delta']    = jnp.full_like(x[:,0], (7./5)  ).reshape(-1,1)  #-- hospitalzied period is something like 5 days or 5/7th of a week
        dynamics['exposed']  = jnp.full_like(x[:,0], (7./2)  ).reshape(-1,1)  #-- exposed period is something like 2 days or 2/7th of a week
        
        #dynamics["trt"]       = jnp.array(x[:,4]).reshape(-1,1)

        dynamics["trt"]        = jnp.mean(VE)
        
        ts          = jnp.linspace(-1,T, T+1)
        cincidencef = partial(compute_incidence,T=T,ts=ts,beta_cov=None,X=None,errs=None, trt=dynamics["trt"])
        incident_cases, incident_ilis, incident_hosps = jax.vmap(cincidencef)(initial_conditions
                                                               ,dynamics["R0"]
                                                               ,dynamics["gamma"]
                                                               ,dynamics["phi"]
                                                               ,dynamics["alpha"]
                                                               ,dynamics["delta"]
                                                               ,dynamics["exposed"])

        #--reshape
        incident_cases = incident_cases.reshape(-1,1,T)
        incident_ilis  = incident_ilis.reshape( -1,1,T)
        incident_hosps = incident_hosps.reshape(-1,1,T)
        
        #--hospitalizations----------------------------------------------------------------------------------------------------
        expected_value_hosps = (N*incident_hosps)
        expected_value_hosps = expected_value_hosps.reshape(NSAMPLES,1,T) #<- 1 over seasons
        
        present  = ~jnp.isnan(hosps)
        ll_hosps = dist.Poisson(expected_value_hosps+eps ).mask(present).log_prob(hosps.reshape(1,nseasons,T))

        hosps_influence = jnp.sum( jnp.sum( ll_hosps, axis=-1), axis=-1)

        #--ILIs-------------------------------------------------------------------------------------------------------------
        if ili_n is not None and ili_i is not None:
            present_ili_n = ~jnp.isnan(ili_n)
            present_ili_i = ~jnp.isnan(ili_i)
            present_ili   = present_ili_n*present_ili_i

            a       = 100*incident_ilis
            b       = 100*(1-incident_ilis)
            ll_ilis = dist.BetaBinomial( a,b, jnp.where(jnp.isnan(ili_n),1,ili_n)).mask(present_ili_i).log_prob(ili_i.reshape(1,nseasons,T))

            ili_influence = jnp.sum( jnp.sum( ll_ilis, axis=-1), axis=-1)
        else:
            ili_influence = 0.

        return -1*(hosps_influence + ili_influence)
    
    def estimate_dynamics(cases,hosps,ili_n,ili_i,N):
        from pymoo.core.problem import Problem
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.optimize import minimize
        
        class SIRproblem(Problem):
            def __init__(self, cases=None, hosps=None, ili_i=None, ili_n=None ,N=None, **kwargs):
                if cases is not None:
                    self.cases = cases+1
                else:
                    self.cases = None
                    
                self.hosps = hosps+1
                self.ili_n = ili_n
                self.ili_i = ili_i
                
                self.N     = N

                nseasons,T    = hosps.shape
                self.nseasons = nseasons
                self.T        = T

                super().__init__( #S0,R0,
                                 n_var          = 1+1+2     #--Number of parameters
                                 , n_obj        = 1           #--Number of objective functions
                                 , n_ieq_constr = 0
                                 , n_eq_constr  = 0      
                                 , xl           = [0.90]*1 + [1 ]  + [0]*2  #+  [1-0.80]*1   
                                 , xu           = [1]*1    + [10]  + [2]*2  #+  [1-0.20]*1   
                )

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = loglikelihood(self,x,cases = self.cases, hosps=self.hosps, ili_n = self.ili_n,ili_i = self.ili_i)
        
        problem = SIRproblem(cases=cases,hosps=hosps, ili_n=ili_n,ili_i=ili_i, N = N)
                         
        algorithm = GA(pop_size=10000,eliminate_duplicates=True)
        res = minimize(problem,
                       algorithm,
                       seed    = 20200320,
                       verbose = False)
        thetas       = res.X
        return thetas

    if ili_n is not None and ili_i is not None:
        thetas = [ estimate_dynamics(cases=None,hosps=x,ili_n=y,ili_i=z,N=N) for x,y,z in zip(hosps,ili_n,ili_i)]
    else:
        thetas = [ estimate_dynamics(cases=None,hosps=x,ili_n=None,ili_i=None,N=N) for x in hosps]
        
    dynamics = {}
    for n,x in enumerate(["R0","phi","alpha"]):
        dynamics[x] = thetas[0][1+n].reshape(1,1,1) 
    dynamics["vacc"]     = jnp.full_like(dynamics["R0"] , 1./2)
    dynamics["gamma"]    = jnp.full_like(dynamics["R0"] , 3./7)
    dynamics["delta"]   = jnp.full_like( dynamics["R0"] , 7./5)
    dynamics["exposed"] = jnp.full_like( dynamics["R0"] , 7./2)

    S0     = (1-dynamics["vacc"].squeeze())*thetas[0][0]
    S0t    = (dynamics["vacc"].squeeze())*thetas[0][0]
    I0     = 1. - (S0+S0t)
    H0     = 0
    
    initsa = jnp.array([S0,S0t,0,0,I0,H0,0, 0,0,0])
    initial_conditions = initsa.reshape(1,1,10)
    
    def expfunc(x,omega):
        return (10**-6)*jnp.eye(x.shape[0]) + omega*jnp.minimum(x[:,None],x[None,:])
    
    def mcmcmodel(cases,hosps,ili_n,ili_i,X,VE,N,seed="",estimate=None,dynamics_input=None,initial_conditions=None):
        eps = 10**-12
        if cases is not None:
            cases = cases+1
        if hosps is not None:
            hosps = hosps+1
        
        if cases is not None:
            ntypes,nseasons,T = cases.shape
        if ili_n is not None:
            ntypes,nseasons,T = ili_n.shape
        if hosps is not None:
            ntypes,nseasons,T = hosps.shape

        #--covariate data X is types,sessons,t,covs
        ncovs = X.shape[-1]
            
        #-----BEGIN------------------------------------------------------------
        #--plates
        type_plate   = numpyro.plate("type"    , ntypes    , dim=-3)
        season_plate = numpyro.plate("season"  , nseasons  , dim=-2)
        time_plate   = numpyro.plate("time"    , T         , dim=-1)

        #--begin dynamics
        dynamics              = dynamics_input.copy()
        dynamics["R0"]        = jnp.log(dynamics.get("R0")   )
        dynamics["phi"]       = logit(dynamics.get("phi")    )
        dynamics["alpha"]     = jnp.log(dynamics.get("alpha"))

        dynamics["trt"]       = VE #<--we are setting here the MWR data
       
        N    = N
        with type_plate:
            s_R0    = numpyro.sample("s_r0", dist.HalfCauchy(10)) 
            R0      = numpyro.sample( "r0", dist.Normal(dynamics["R0"]     , s_R0 ) )

            s_phi   = numpyro.sample("s_phi", dist.HalfCauchy(10)) 
            phi     = numpyro.sample( "phi", dist.Normal(dynamics["phi"]   , s_phi ))

            s_alpha = numpyro.sample("s_alpha", dist.HalfCauchy(10)) 
            alpha   = numpyro.sample("alpha", dist.Normal(dynamics["alpha"], s_alpha ))

            dynamics["trt"] = VE

            #--susceptability
            S   = jnp.sum(initial_conditions[:, 0, :1+1],axis=-1).reshape(-1,1,1)
            S   = numpyro.sample("S", dist.Normal( logit(S), 10 ))
            S   = expit(S)
               
        with type_plate:
            Gp1mean = numpyro.sample("Gp1mean", dist.Beta(1,1))
            Gp1err  = numpyro.sample("Gp1err" , dist.LogNormal(0,1))

            with season_plate:
                GP1_season_   = numpyro.sample("GP1_season_", dist.LogNormal( Gp1mean, Gp1err ) )

                
        #-- residual termd for log R0
        Ks_season        = jax.vmap(jax.vmap(lambda omega: expfunc( jnp.arange(T),omega)))(GP1_season_)
        deltas_season = numpyro.sample("deltas_season", dist.MultivariateNormal(0, Ks_season))
        deltas_season = deltas_season.at[...,0].set(0)
        deltas_season = deltas_season.reshape(ntypes,nseasons,T)
        deltas_season = jnp.clip( deltas_season, -jnp.inf,0 )
        
        #--covariate parameters
        beta_covs     = numpyro.sample("beta_covs__m", dist.Normal(0,1).expand([ntypes,ncovs]))
        beta_covs     = beta_covs.reshape(ntypes,1,ncovs)
            
        dynamics["alpha"]     = jnp.exp(alpha).reshape(ntypes,1,1)
        dynamics["phi"]       = expit(phi).reshape(ntypes,1,1)
        
        grand_R0s             = jnp.exp(R0).reshape(ntypes,1,1)

        dynamics["R0"]        = grand_R0s
        
        dynamics["vacc"]      = jnp.full_like(grand_R0s, 1./2)
        
        dynamics["gamma"]     = jnp.full_like(grand_R0s, 3./7)
        
        dynamics["delta"]     = jnp.full_like(grand_R0s, 7./5)
        dynamics["exposed"]   = jnp.full_like(grand_R0s, 7./2)
            
        sigmas_R0s        = numpyro.sample("R0_sigma", dist.HalfCauchy(1) )
            
        #--observational process
        stderr            = numpyro.sample("stderr"     , dist.Gamma(1,1))
        stderr_ili        = numpyro.sample("stderr_ili" , dist.HalfCauchy(1.))

        with type_plate:
            with season_plate:
                log_R0s   = numpyro.sample("R0s_log"       , dist.Normal( jnp.log(grand_R0s)   ,sigmas_R0s ))
                R0s       = numpyro.deterministic("R0",jnp.clip(jnp.exp(log_R0s),0,10))
               
                def create_transmission_rate(logr0,x,b):
                    return jnp.exp(logr0 + x.dot(b.reshape(ncovs,1)) )
                transmission_rate = jax.vmap(jax.vmap( create_transmission_rate, in_axes=(0,0,None) ))(log_R0s,X,beta_covs)
                numpyro.deterministic("transmission_rate", transmission_rate)

                S0   = (1-dynamics["vacc"])*S 
                S0t  =   (dynamics["vacc"])*S 
                I0   = 1. - (S0+S0t)

                Z                  = jnp.broadcast_to( jnp.zeros((ntypes,1,1)), S0.shape)
                initial_conditions = jnp.concatenate([S0,S0t,Z,Z,I0,Z,Z,  I0,Z,Z],axis=-1)

                ts          = jnp.linspace(-1,T, T+1)
                cincidencef = partial(compute_incidence,T=T,ts=ts,errs=None)
                incident_cases, incident_ilis, incident_hosps = jax.vmap(jax.vmap(cincidencef,in_axes=(None,) + (0,) + (None,)*5 + (0,) + (None,) + (0,) ))(initial_conditions
                                                                                                                                                  ,R0s
                                                                                                                                                  ,dynamics["gamma"]
                                                                                                                                                  ,dynamics["phi"]
                                                                                                                                                  ,dynamics["alpha"]
                                                                                                                                                  ,dynamics["delta"]
                                                                                                                                                  ,dynamics["exposed"]
                                                                                                                                                  ,dynamics["trt"]
                                                                                                                                                  ,beta_covs
                                                                                                                                                  ,X)
                                                                                                                                                  
 

                with time_plate:
                    scaling_factor = numpyro.deterministic("scaling_factor", N)

                    if hosps is not None:
                        ys   = logit(hosps/scaling_factor)
                        yhat = jnp.clip(logit(incident_hosps)   ,-30,0)
                        
                        numpyro.deterministic("yhat", yhat)

                        hosps_smooth = jnp.clip(scaling_factor*incident_hosps ,eps,N)
                        numpyro.deterministic("hosps_smooth", hosps_smooth)

                        hosps_plus_delta  = scaling_factor*expit(yhat + deltas_season[:,:,::-1])
                        numpyro.deterministic("hosps_plus_delta", hosps_plus_delta)

                        present = ~jnp.isnan(hosps)
                        with numpyro.handlers.mask(mask=present):
                            numpyro.sample("hosp_obs_{:s}".format(location), dist.NegativeBinomial2(hosps_plus_delta,50) , obs = hosps.reshape(ntypes,nseasons,T) )
                            
                    if ili_n is not None and ili_i is not None:
                        present_ili_n = ~jnp.isnan(ili_n)
                        present_ili_i = ~jnp.isnan(ili_i)

                        incident_cases = numpyro.deterministic("incident_ilis", incident_ilis)
                        
                        with numpyro.handlers.mask(mask=present_ili_n*present_ili_i):
                            a = stderr_ili*incident_ilis
                            b = stderr_ili*(1-incident_ilis)
                            numpyro.sample("ili_obs_{:s}".format(location), dist.BetaBinomial( a,b, jnp.where(jnp.isnan(ili_n),1,ili_n)), obs = ili_i.reshape(ntypes,nseasons,T) )

    #--shape diagnosis
    rng_key = jax.random.PRNGKey(0)
    # Wrap model in a seed and trace context to capture shapes
    with trace() as tr:
        seeded_model = seed(mcmcmodel, rng_key)  # Seed the model with RNG key
        seeded_model(cases                      = cases
                     , hosps                    = hosps
                     , ili_n                    = ili_n
                     , ili_i                    = ili_i
                     , X                        = X
                     , VE                       = VE 
                     , N                        = N
                     , dynamics_input           = dynamics
                     , initial_conditions       = initial_conditions
                     ,estimate                  = True
                     ,seed                      = str(location))
    print(format_shapes(tr))
                        
    num_warmup  = 1*10*10**3
    num_samples = 1*10*10**3
    
    guide = AutoNormal(mcmcmodel, init_loc_fn=init_to_median)

    optimizer  = numpyro.optim.Adam(step_size=0.001)
    svi        = SVI(mcmcmodel, guide, optimizer, loss=Trace_ELBO())
    svi_result = svi.run(jax.random.PRNGKey(0)
                         , 2*10**4
                         , cases                     = cases
                         , hosps                     = hosps
                         , ili_n                     = ili_n
                         , ili_i                     = ili_i
                         , X                         = X
                         , VE                        = VE 
                         , dynamics_input            = dynamics
                         , initial_conditions        = initial_conditions
                         , N                         = N
                         , estimate                  = True
                         , seed                      = str(location) )

    predictive  = Predictive(mcmcmodel, guide=guide, params = svi_result.params, return_sites = ["hosps_plus_delta","hosps_smooth","incident_ilis","transmission_rate","beta_covs__m"], num_samples=10*10**3)

    #--look at 
    predictions = predictive(rng_key
                             , cases               = cases
                             , hosps               = hosps
                             , ili_n               = ili_n
                             , ili_i               = ili_i
                             , X                   = X
                             , VE                  = VE 
                             , N                   = N
                             , dynamics_input      = dynamics
                             , initial_conditions  = initial_conditions
                             , estimate            = False
                             , seed = str(location))

    concentration = 50
    yhats =  dist.NegativeBinomial2(predictions["hosps_plus_delta"],concentration ).sample(jax.random.PRNGKey(2020), (1,)).squeeze()

    return None, predictions, yhats

if __name__ == "__main__":

    #--data set of populations (contains all FIPS)
    pops            = pd.read_csv("./data_sets/locations.csv")
    
    #--incident hospitalizations dataset
    inc_hosps = pd.read_csv("./data_sets/target-hospital-admissions.csv")

    pct_hosps_reporting = pd.read_csv("./data_sets/pct_hospital_reporting.csv")
    
    #--subset by only information after 09-01
    inc_hosps = inc_hosps.loc[ (inc_hosps["date"]>="2021-10-09")  ]

    #--ILI data
    ili_data = pd.read_csv("./data_sets/ilidata.csv")
    ili_data["week"] = [ int(str(x)[-2:]) for x in ili_data.epiweek]

    #--lab data
    lab_data = pd.read_csv("./data_sets/clinical_and_public_lab_data__formatted.csv")

    ili_augmented = lab_data.merge(ili_data, on = ["state","epiweek","week"])

    #--CUT ALL OF THIS DATA TO BEFORE THE BEGINNING OF LAST SEASON.
    start_of_2324_season = Week(2023,40).startdate().strftime("%Y-%m-%d")
    inc_hosps            = inc_hosps.loc[ inc_hosps.date < start_of_2324_season]
    ili_augmented        = ili_augmented.loc[ ili_augmented.epiweek <202340 ]

    #--remove weeks betweeen 20 and 40
    def compute_forecasts(location, time, peak_value=None, peak_time=None):
        import jax.numpy as jnp
        from epiweeks import Week
        from datetime import datetime, timedelta

        #--take state only
        state_hosps     = inc_hosps.loc[inc_hosps.location==location]
        population_size = int(pops.loc[pops.location==location,"population"])

        state_ili             = ili_augmented.loc[ (ili_augmented.location==location) & (ili_augmented.season!="offseason")]
        state_ili["end_date"] = [Week(int(str(x)[:3+1]),int(str(x)[-2:])).enddate() for x in state_ili.epiweek ] 

        state_ili["num_ili_scaled"] = state_ili.num_ili*(state_ili.percent_positive/100)

        state_pct_hosp = pct_hosps_reporting.loc[pct_hosps_reporting.location==location]
        
        dates = state_hosps["date"].unique()

        #--sort by earliest to latest
        location_data = state_hosps.sort_values(["date"])
        
        #--add_season
        def add_season(x):
            from epiweeks import Week
            epiweek = int(Week.fromdate( datetime.strptime(x["date"], "%Y-%m-%d")).cdcformat())

            y,wk = int(str(epiweek)[:3+1]), int(str(epiweek)[-2:])
            if wk>=40:
                start_of_season = int("{:04d}40".format(y))
                end_of_season   = int("{:04d}20".format(y+1))
                season          = "{:04d}/{:04d}".format(y,y+1)
            else:
                start_of_season = int("{:04d}40".format(y-1))
                end_of_season   = int("{:04d}20".format(y))
                season          = "{:04d}/{:04d}".format(y-1,y)
            x["season"]          = season
            x["start_of_season"] = start_of_season
            x["end_of_season"]   = end_of_season
            x["epiweek"]         = epiweek
            return x
        location_data  = location_data.apply(add_season,1)

        state_pct_hosp         = state_pct_hosp.apply(add_season,1)
        state_pct_hosp["week"] = [int(str(x)[-2:]) for x in state_pct_hosp.epiweek.values]
        
        ordered_weeks = list(np.arange(40,52+1)) + list(np.arange(1,20+1))
        
        ili__N = pd.pivot_table(index=["season"], columns = ["week"] , values = ["num_patients"]   , data = state_ili, dropna=False)
        ili__I = pd.pivot_table(index=["season"], columns = ["week"] , values = ["num_ili_scaled"] , data = state_ili, dropna=False)

        ili__N.columns = [y for (x,y) in ili__N.columns]
        ili__I.columns = [y for (x,y) in ili__I.columns]

        state_pct_hosp = pd.pivot_table(index=["season"], columns = ["week"] , values = ["pct_hosp"]   , data = state_pct_hosp, dropna=False)
        state_pct_hosp.columns = [y for (x,y) in state_pct_hosp.columns]
        
        subset = location_data.copy()
        
        subset["week"]      = [int(str(x)[-2:]) for x in subset.epiweek.values]
        season_time         = pd.pivot_table( index=["season"], columns = ["week"], values=["value"], data = subset, dropna=False )
        season_time.columns = [y for (x,y) in season_time.columns]

        #--add columns if missing a week
        for x in ordered_weeks:
            if x not in season_time:
                season_time[x] = np.nan
        season_time = season_time[ [x for x in ordered_weeks]  ]

        ili__N = ili__N[ordered_weeks]
        ili__I = ili__I[ordered_weeks]
        state_pct_hosp = state_pct_hosp[ordered_weeks]

        all_seasons = sorted( [ "{:d}/{:d}".format(x,x+1) for x in np.arange(2015,2023+1) ])
        
        for season in all_seasons:
            if season not in season_time.index:
                season_time.loc[season] = np.nan
            if season not in ili__N.index:
                ili__N.loc[season] = np.nan
            if season not in ili__I.index:
                ili__I.loc[season] = np.nan
            if season not in state_pct_hosp.index:
                state_pct_hosp.loc[season] = np.nan

                
               
        season_time = season_time.reindex(all_seasons)
        ili__N = ili__N.reindex(all_seasons)
        ili__I = ili__I.reindex(all_seasons)
        state_pct_hosp = state_pct_hosp.reindex(all_seasons)

        #--Remove pandmeic season
        ili__N.loc["2020/2021"]=np.nan 
        ili__I.loc["2020/2021"]=np.nan
        state_pct_hosp.loc["2020/2021"]=np.nan 
        
        T = season_time.shape[-1]

        season_time_augmented = season_time / state_pct_hosp
        print(season_time_augmented)
        
        season_hosps = season_time_augmented.to_numpy()
        print(season_time_augmented.to_numpy()[-1,:])

        #--WEATHER DATA
        weather_data = pd.read_csv("./data_sets/weekly_weather_data.csv")
        
        time_data = state_ili[["location","season","end_date","year","week"]].drop_duplicates()
        time_data["end_date"] = [datetime.strftime(x,"%Y-%m-%d") for x in time_data.end_date.values]
        
        location_weather = weather_data.merge( time_data
                                , left_on  = ["location","enddate","year","week"]
                                , right_on = ["location","end_date","year","week"] )

        
        #--weather covariates
        #--STATE LEVEL AVG TEMP 
        temp = pd.pivot_table( index="season", columns = ["week"], values = "tavg", data = location_weather, dropna=False ).reset_index()
        temp = temp.set_index("season").interpolate(axis=1)
        temp.loc["2023/2024"] = np.nan
        temp = temp.reset_index()
    
        temp = temp[ ["season"] + list(np.arange(40,52+1)) + list(np.arange(1,20+1)) ] 

        #--prepare TEMP MATRIX
        temp_data = temp.iloc[:,1:].to_numpy()

        #--replace the rest of the season without data with an average of past seasons
        cutoff           = np.min(np.where(np.isnan(season_hosps[-1,:])))
        avg_without_data = np.nanmean(temp_data[:-1,cutoff:],0) #<-avg over seasons

        temp_data[-1,cutoff:] = avg_without_data
        
        def standardize(x):
            X =  (x - np.nanmean(x,1).reshape(-1,1))/ np.nanstd(x,1).reshape(-1,1)

            #--smooth
            for n,row in enumerate(X):
                X[n,:] = gaussian_filter1d(row,2)
            
            X = np.append( X[:,0].reshape(-1,1) , X, axis=1 ) #adding a repeat bc we start integration one week before

            X = X[np.newaxis,...]
            return X

        temp_data_centered = standardize(temp_data)
        #------------------------------------------------------------------------------------------------------------------------
        #--PRESSURE WHICH IS RELATED TO RELATIVE HUMIDITY
        pres                  = pd.pivot_table( index="season", columns = ["week"], values = "pavg", data = location_weather, dropna=False ).reset_index()
        pres                  = pres.set_index("season").interpolate(axis=1)
        pres.loc["2023/2024"] = np.nan
        pres                  = pres.reset_index()
        pres                  = pres[ ["season"] + list(np.arange(40,52+1)) + list(np.arange(1,20+1)) ] 

        #--prepare PRES MATRIX
        #nseasons  = 13
        pres_data = pres.iloc[:,1:].to_numpy()

        #--replace the rest of the season without data with an average of past seasons
        cutoff           = np.min(np.where(np.isnan(season_hosps[-1,:])))
        avg_without_data = np.nanmean(pres_data[:-1,cutoff:],0) #<-avg over seasons

        pres_data[-1,cutoff:] = avg_without_data
        
        pres_data_centered = standardize(pres_data)

        X = np.stack( [temp_data_centered, pres_data_centered], axis=-1 )

        mmwr_ve = pd.read_csv("./data_sets/VE_mmwr.csv")
        mmwr_ve              = mmwr_ve.iloc[2:]
 
        mmwr_ve = pd.concat([ pd.DataFrame({"link":["madeup"], "season":["2023/2024"],"ve":[np.mean( mmwr_ve.ve.values )]} ) , mmwr_ve]) 
        ve      = mmwr_ve["ve"].to_numpy().reshape(-1,1)[np.newaxis,...]
        ve      = (1.-ve)

        #--no data in this season, so assume the average
        ili__I.iloc[-1,:] = ili__I.mean(0)
        ili__N.iloc[-1,:] = ili__N.mean(0)
        
        theta, posterior_samples,yhats = model( cases=None
                                                        , ili_n            = ili__N.to_numpy()[jnp.newaxis,...]
                                                        , ili_i            = ili__I.to_numpy()[jnp.newaxis,...]
                                                        , hosps            = season_time_augmented.to_numpy()[jnp.newaxis,...]
                                                        , X                = X
                                                        , N                = population_size
                                                        , VE               = ve
                                                        , location         = location)

        peak_time      = jnp.argmax( yhats[:,-1,:], axis=-1  )#--the -1 is the most recent season
        peak_intensity = jnp.max(    yhats[:,-1,:], axis=-1  )#--the -1 is the most recent season
        
        #--STORE DATA-----------------------------------------------------
        #---extract quantiles
        quantiles          = np.append(np.append([0.01,0.025],np.arange(0.05,0.95+0.05,0.05)), [0.975,0.99])
        
        #--WEEKLY INCIDENCE DATA------------------------------------------------------------------------------
        #weekly_times     =  np.percentile( fitted_samples["hosps_hat"][:,0,-1,:], quantiles*100, axis=0) #--the -1 is the most recent season
        weekly_times            = np.percentile(yhats[:,-1,:], quantiles*100, axis=0) #--the -1 is the most recent season
        weekly_peak_times       = np.percentile( peak_time     , quantiles*100, axis=0)      
        weekly_peak_intensities = np.percentile( peak_intensity, quantiles*100, axis=0) 
        
        def generate_epiweek_end_dates(start_year, start_week, end_year, end_week):
            end_dates = []
            current_week = Week(start_year, start_week)
            end_week_obj = Week(end_year, end_week)

            while current_week <= end_week_obj:
                # Calculate the Sunday (end of the week)
                end_dates.append(current_week.enddate())
                # Move to the next week
                current_week = current_week + 1

            return end_dates

        # Define the start and end epiweeks for the 2024/2025 season
        start_year, start_week = 2023, 40  
        end_year, end_week     = 2024, 20  

        reference_date         = Week(start_year,start_week).enddate()
        
        # Generate and print all epiweek end dates for the 2024/2025 influenza season
        timepoints = generate_epiweek_end_dates(start_year, start_week, end_year, end_week)
        
        #--add data to dictionary
        forecast_data = {"reference_date"  :[]
                         ,"horizon"        :[]
                         ,"target_end_date":[]
                         ,"output_type_id" :[]
                         ,"value"          :[]}
        for forecast_time,d in zip(timepoints, weekly_times.T):
            fmt = "%Y-%m-%d"
            
            forecast_data["reference_date"].extend( [reference_date.strftime(fmt)]*23 )

            week_from_reference = int((forecast_time - reference_date).days/7)
            
            forecast_data["horizon"].extend( [week_from_reference]*23 )

            ted = Week.fromdate(forecast_time).enddate().strftime(fmt)
            forecast_data["target_end_date"].extend([ted]*23)

            forecast_data["output_type_id"].extend( ["{:0.3f}".format(x) for x in quantiles] )
            forecast_data["value"].extend( [ int(x) for x in np.floor(d)] )
            
        weekly_forecast_data = pd.DataFrame(forecast_data)
        weekly_forecast_data["location"]    = location
        weekly_forecast_data["output_type"] = "quantile"
        weekly_forecast_data["target"]      = "wk inc flu hosp"

        columns = ["reference_date","target","horizon","target_end_date","location","output_type","output_type_id","value"]
        weekly_forecast_data = weekly_forecast_data[columns]

        #--PEAK time
        
        #--add data to dictionary
        ntimepoints = len(timepoints)
        forecast_data = {"date":[], "location":[], "location_name":[], "peak_value":[], "peak_time":[]  }
        forecast_data = {"reference_date":[]
                         ,"output_type_id":[]
                         ,"value":[]}
        
        forecast_data["reference_date"]  = [reference_date]*ntimepoints
        forecast_data["location"]        = [location]*ntimepoints
        forecast_data["output_type_id"]  = timepoints
        forecast_data["value"]           = [ float(np.mean(peak_time==x))   for x in np.arange(33)] 
        
        peak_forecast_time_data                    = pd.DataFrame(forecast_data)
        peak_forecast_time_data["horizon"]         = np.nan
        peak_forecast_time_data["target_end_date"] = np.nan
        peak_forecast_time_data["output_type"]     = "pmf"
        peak_forecast_time_data["target"]          =  "peak week inc flu hosp"

        peak_forecast_time_data = peak_forecast_time_data[columns]
        
        #--PEAK INCIDIENCE
        nquantiles = 23
        forecast_data = {"date":[], "location":[], "location_name":[], "peak_value":[], "peak_time":[]  }
        forecast_data = {"reference_date":[]
                         ,"output_type_id":[]
                         ,"value":[]}
        
        forecast_data["reference_date"]  = [reference_date]*nquantiles
        forecast_data["location"]        = [location]*nquantiles
        forecast_data["output_type_id"]  = ["{:0.3f}".format(x) for x in quantiles]
        forecast_data["value"]           = [int(x) for x in weekly_peak_intensities ]
        
        peak_forecast_intensity_data                    = pd.DataFrame(forecast_data)
        peak_forecast_intensity_data["horizon"]         = np.nan
        peak_forecast_intensity_data["target_end_date"] = np.nan
        peak_forecast_intensity_data["output_type"]     = "quantile"
        peak_forecast_intensity_data["target"]          = "peak inc flu hosp"

        peak_forecast_intensity_data = peak_forecast_intensity_data[columns]
        
        thisweek = Week.thisweek().enddate().strftime("%Y-%m-%d")
        
        weekly_forecast_data.to_csv("./forecasts/with_signals/weekly_forecasts__{:s}__{:s}.csv".format(location,thisweek))
        peak_forecast_time_data.to_csv("./forecasts/with_signals/peak_forecasts_time__{:s}__{:s}.csv".format(location,thisweek))
        peak_forecast_intensity_data.to_csv("./forecasts/with_signals/peak_forecasts_intensity__{:s}__{:s}.csv".format(location,thisweek))

        all_forecasts = pd.concat([weekly_forecast_data,peak_forecast_time_data, peak_forecast_intensity_data])
        all_forecasts.to_csv("./forecasts/with_signals/all_forecasts__{:s}__{:s}.csv".format(location,thisweek))

        pickle.dump(posterior_samples, open("./time_dep_transmission_rate/with_signals_posterior_samples.pkl","wb"))
        
        return None

    #---------------------------------------------------------------------------------------------------
    # If the User wants to run only the US forecast presented in the paper then uncomment this code
    for location in sorted(inc_hosps.location.unique()):
        if location =="US":
            compute_forecasts(location=location,time=None,peak_time=None,peak_value=None)
    #---------------------------------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------------------------------
    # If the user wants to run all states plus the US forecasts than un comment the below code
    #---------------------------------------------------------------------------------------------------
    # dask.config.set({
    #     "distributed.worker.memory.target"   : 0.6,   # Triggers memory management at 60% usage
    #     "distributed.worker.memory.spill"    : 0.7,    # Moves data to disk at 70% usage
    #     "distributed.worker.memory.pause"    : 0.8,    # Pauses tasks at 80% usage
    #     "distributed.worker.memory.terminate": 0.9}) # Restarts worker at 90% usage
    
    # from dask.distributed import Client

    # client = Client(n_workers=10,threads_per_worker=1, processes=True,timeout="120s", heartbeat_interval="30s")
    # client.restart()

    # def try_forecast(location):
    #     try:
    #         compute_forecasts(location=location,time=None,peak_time=None,peak_value=None)
    #     except:
    #         print("Failed = {:s}".format(location))
            
    # tasks   = [ delayed(try_forecast)(location=location) for location in sorted(inc_hosps.location.unique()) ]

    # try:
    #     results = compute(*tasks)
    # except Exception as e:
    #     print(f"Error during computation: {e}")
    # client.close()
