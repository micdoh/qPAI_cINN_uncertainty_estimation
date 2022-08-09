import qPAI_cINN_uncertainty_estimation.config as c


def config_string(file):
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    for v in dir(c):
        if v[0] == '_':
            continue
        s = eval(f'c.{v}')
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "="*80 + "\n"

    with open(file, 'w') as f:
        f.write(config_str)

    return config_str
