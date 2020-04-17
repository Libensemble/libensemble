def rc(**kargs):
    """Runtime configuration options.

    Parameters
    ----------
    aposmm_optimizers : string or list of strings
        Select the aposmm optimizer/s (to prevent all options being imported).


    """
    for key in kargs:
        if not hasattr(rc, key):
            raise TypeError("unexpected argument '{0}'".format(key))
    for key, value in kargs.items():
        setattr(rc, key, value)


rc.aposmm_optimizers = None
__import__('sys').modules[__name__ + '.rc'] = rc
