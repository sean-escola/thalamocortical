import numpy as np
import dill
import getopt
import sys

from my_lbfgs import build_model, do_adam, do_lbfgs

np.set_printoptions(linewidth=200)

settings = dict(
    # network parameters
    tau = 0.01,  # neural time constant (s)
    Ns = [100, 300, 500],  # network sizes, must be an iterable
    n_fracs = [0.03, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5],  # rank of U and V as fraction of N, must be an iterable
    K = 5,  # number of repeats per condition

    # random seeds
    np_seed = 1234,

    # penalty for solutions that don't decay
    alpha = 1e5,

    # smoothness constraint
    betas = [0.01],  # for all the stats over N and n
    # looped_beta_params = dict(betas=[0, 0.001, 0.01, 0.1, 1.0], N=500, n_frac=0.1)  # for stats over beta

    # learning
    adam_epochs = 30,
    learning_rate = 0.01,
    lbfgs_iters = 200,
    print_interval = 10,
    verbosity = 1,

    # output filename
    filename = 'out.dill',
)

def main(argv=sys.argv):
    ints = {
        '-k': 'K',
        '-p': 'print_interval',
        '-L': 'lbfgs_iters',
        '-A': 'adam_epochs',
        '-r': 'np_seed',
        '-v': 'verbosity'
    }
    floats = {
        '-a': 'alpha',
        '-T': 'tau',
        '-l': 'learning_rate',
    }
    int_arrays = {
        '-N': 'Ns',
    }
    float_arrays = {
        '-F': 'n_fracs',
        '-b': 'betas',
    }
    strings = {
        '-f': 'filename',
    }
    
    opts, _ = getopt.getopt(argv[1:], 'k:p:L:A:r:a:b:T:l:N:F:f:v:')
    for o, a in opts:
        if o in ints:
            settings[ints[o]] = int(a)
        elif o in floats:
            # try:
                settings[floats[o]] = float(a)
            # except ValueError:
            #     settings[floats[o]] = float(eval(a))
        elif o in int_arrays:
            settings[int_arrays[o]] = list(np.fromstring(a, sep=' ', dtype=int))
        elif o in float_arrays:
            settings[float_arrays[o]] = list(np.fromstring(a, sep=' ', dtype=float))
        elif o in strings:
            settings[strings[o]] = a

    if settings['verbosity']:
        print('Settings are:')
        for k in settings:
            print(f'   `{k}` : `{settings[k]}`')

    np.random.seed(settings['np_seed'])

    models = [build_model(N, settings['n_fracs'], settings['K'], settings['tau'], settings['alpha'], settings['betas']) for N in settings['Ns']]

    outs = []
    for i in range(len(settings['Ns'])):
        print('')
        print(f'------------------------------------------------------')
        print(f"N: {settings['Ns'][i]}")
        param_adam, l_adam = do_adam(models[i], settings['adam_epochs'], learning_rate=settings['learning_rate'], print_interval=settings['print_interval'], verbosity=2)
        param, l_lbfgs, results = do_lbfgs(models[i], settings['lbfgs_iters'], print_interval=settings['print_interval'], param=param_adam, losses=l_adam[:-1], verbosity=3, tolerance=1e-5)
        outs.append((param, l_lbfgs, results))
    with open(settings['filename'], 'wb') as f:
        dill.dump((outs, models), f)

if __name__ == "__main__":
    main()
