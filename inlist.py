
def prep_inlist(template, dest, subst):
    with open(template) as ftemp:
        with open(dest, 'w') as finl:
            for line in ftemp:
                arr = line.split()
                if len(arr)==0: continue
                p = arr[0]
                if p in subst:
                    finl.write(p + ' = ' + subst[p] + '\n')
                else:
                    finl.write(line)


