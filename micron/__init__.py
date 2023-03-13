from importlib import import_module
import os
import sys


HOME = os.environ['HOME']
MICRON_CACHE = os.path.join(HOME, '.cache', 'micron')

os.environ["WANDB_DISABLED"] = "TRUE"

def run(*args):
    if len(args) == 0:
        console = True
        args = sys.argv[1:]
    else:
        console = True

    def print_usage():
        if console:
            print(f"""Usage:\t{args[0]} --help | {{package.module.Class}} {{init_kvargs}} {{method_name}}""")
        else:
            print(f"""Usage:\n{__name__}.run("--help")  |\n{__name__}.run("{{package.module.Class}}, {{init_kvarg, init_kvarg, ...}}, {{method_name}}")""")

    if len(args) == 0:
        print_usage()
        if console:
            sys.exit(1)
        else:
            raise ValueError

    if args[0] == '--help':
        print_usage()
        if len(args) > 1:
            if console:
                sys.exit(1)
            else: 
                raise ValueError
        else:
            if console:
                sys.exit(0)
            else:
                return

    parts = args[0].split('.')
    class_name = parts[-1]
    module_name = '.'.join(parts[:-1])
    try:
        module_ = import_module(module_name)
        class_ = getattr(module_, class_name)
    except Exception as e:
            print_usage()
            raise(e)
    kveq_list = args[1:-1]
    method_name = args[-1]
    kvargs = [kveq.split('=') for kveq in kveq_list]
    kwargs = {k: v for k, v in kvargs}
    instance_ = class_(**kwargs)
    try: 
        method_ = getattr(instance_, method_name)
    except Exception as e:
        print_usage()
        raise(e)
    _ = method_()
    return _
