import numpyro
import jax
import arviz as az

def get_ppc(model, mcmc, rng_key_seed=0, **kwargs):
    """MCMCサンプルを使った事後分布のサンプルを得る。

    Args:
        model (_type_): _description_
        mcmc (_type_): _description_
        rng_key_seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    mcmc_samples = mcmc.get_samples()

    predictive = numpyro.infer.Predictive(model, mcmc_samples)

    ppc_samples = predictive(jax.random.PRNGKey(rng_key_seed), **kwargs)

    idata_ppc = az.from_numpyro(mcmc, posterior_predictive=ppc_samples)
    return ppc_samples, idata_ppc