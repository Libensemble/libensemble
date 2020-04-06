def rc(**kargs):
    """Runtime configuration options.

    Parameters
    ----------
    aposmm_optimizer : string
        Set the aposmm optimizer (to prevent all options being imported).


    """
    for key in kargs:
        if not hasattr(rc, key):
            raise TypeError("unexpected argument '{0}'".format(key))
    for key, value in kargs.items():
        setattr(rc, key, value)


rc.aposmm_optimizer = True
__import__('sys').modules[__name__ + '.rc'] = rc
