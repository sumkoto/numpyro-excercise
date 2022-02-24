
import numpyro
import jax

import matplotlib.pyplot as plt
import arviz as az

def run_mcmc_inference(model, rng_key_seed=0, num_warmup=1000, num_samples=5000, num_chains=4, **kwargs):
    """定義したモデルに対してmcmcによる推論を実行する。

    Args:
        model (_type_): _description_
        rng_key_seed (int, optional): _description_. Defaults to 0.
        num_warmup (int, optional): _description_. Defaults to 1000.
        num_samples (int, optional): _description_. Defaults to 5000.
        num_chains (int, optional): _description_. Defaults to 4.

    Returns:
        _type_: _description_
    """
    nuts = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

    mcmc.run(jax.random.PRNGKey(rng_key_seed), **kwargs)
    trace = mcmc.get_samples()

    idata = az.from_numpyro(mcmc)
    return mcmc, trace, idata

def check_mccm_inference(idata):
    az.plot_trace(idata)
    plt.show()

    display(az.summary(idata))

    az.plot_posterior(idata)
    plt.show()
