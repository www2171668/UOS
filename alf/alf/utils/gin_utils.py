

import gin.config


def inoperative_config_str(max_line_length=80, continuation_indent=4):
    """Retrieve the "inoperative" configuration as a config string.

    Args:
        max_line_length (int): A (soft) constraint on the maximum length
            of a line in the formatted string.
        continuation_indent (int): The indentation for continued lines.
    Returns:
        A config string capturing all parameter values configured but not
            used by the current program (override by explicit call).
    """
    inoperative_config = {}
    config = gin.config._CONFIG
    operative_config = gin.config._OPERATIVE_CONFIG
    imported_module = gin.config._IMPORTED_MODULES
    for module, module_config in config.items():
        inoperative_module_config = {}
        if module not in operative_config:
            inoperative_module_config = module_config
        else:
            operative_module_config = operative_config[module]
            for key, value in module_config.items():
                if key not in operative_module_config or \
                        value != operative_module_config[key]:
                    inoperative_module_config[key] = value

        if inoperative_module_config:
            inoperative_config[module] = inoperative_module_config

    # hack below
    # `gin.operative_config_str` only depends on `_OPERATIVE_CONFIG` and `_IMPORTED_MODULES`
    gin.config._OPERATIVE_CONFIG = inoperative_config
    gin.config._IMPORTED_MODULES = {}
    inoperative_str = gin.operative_config_str(max_line_length,
                                               continuation_indent)
    gin.config._OPERATIVE_CONFIG = operative_config
    gin.config._IMPORTED_MODULES = imported_module
    return inoperative_str
