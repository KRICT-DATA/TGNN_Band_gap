import pandas
from pymatgen import MPRester


def download_mp_data(form, spc_num, id, pbe, mbj, gw, path):
    mpr = MPRester('z9dAjdwO95SWgsUZ')
    data = mpr.query({'pretty_formula': form}, ['material_id', 'pretty_formula', 'spacegroup', 'band_gap', 'cif'])
    t = None

    for c in data:
        if c['pretty_formula'] == form and c['spacegroup']['number'] == spc_num:
            t = (c['material_id'], c['pretty_formula'], c['spacegroup']['number'], c['band_gap'], pbe, mbj, gw, str(id))
            cif_file = open(path + '/' + c['material_id'] + '.cif', 'w')
            cif_file.write(c['cif'])

    if t is None:
        print(id, form, spc_num)
        return None
    else:
        return t
