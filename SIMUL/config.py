
def parse_conf(fn):
    opt = {}
    with open(fn) as f:
        for line in f:
            text = line[:line.find('#')]
            parts = text.split('=')
            if len(parts) < 2: continue
            name = parts[0].strip()
            arr = parts[1].split(',')
            opt[name] = [a.strip() for a in arr]
            if len(opt[name]) == 1: opt[name] = opt[name][0]
    return opt
