import numpy
import pandas
import random
from pymatgen.core.structure import Structure
from sklearn.svm import SVR
from sklearn.preprocessing import scale
import util.crystal_conv as cc


list_crys = list()
id_target = numpy.array(pandas.read_csv('../data/crystal/nlhm/id_target.csv'))
num_train_ins = int(id_target.shape[0] * 0.8)

for i in range(0, id_target.shape[0]):
    crys = Structure.from_file('../data/crystal/nlhm/' + id_target[i, 0] + '.cif')
    atoms = crys.atomic_numbers
    ratio = crys.formula.split(' ')
    r_avg = 0
    atom_feats_mean = numpy.empty(6)
    atom_feats_std = numpy.empty(6)

    for j in range(0, len(ratio)):
        r_avg += int(''.join([k for k in ratio[j] if not k.isalpha()]))

    r_avg /= len(ratio)

    for j in range(0, len(ratio)):
        r_n = int(''.join([k for k in ratio[j] if not k.isalpha()]))

        atom_feats = cc.mat_atom_feats[atoms[j]-1, :]
        atom_feats_mean += r_n * atom_feats
        atom_feats_std += numpy.square((r_n - r_avg)) * atom_feats

    atom_feat = numpy.hstack([atom_feats_mean, atom_feats_std, crys.volume, id_target[i, 1], id_target[i, 2]])
    list_crys.append(atom_feat)


num_feats = 14
random.shuffle(list_crys)
data = numpy.array(list_crys)
data_x = scale(data[:, :num_feats])
train_data_x = data_x[:num_train_ins, :]
train_data_y = data[:num_train_ins, num_feats]
test_data_x = data_x[num_train_ins:, :]
test_data_y = data[num_train_ins:, num_feats]

cs = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
es = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

best_mae = 1e+8
best_rmse = 1e+8

for c in cs:
    for e in es:
        svr = SVR(C=c, epsilon=e, gamma='auto')
        svr.fit(train_data_x, train_data_y)
        pred = svr.predict(test_data_x)
        mae = numpy.mean(numpy.abs(test_data_y - pred))
        rmse = numpy.sqrt(numpy.mean((test_data_y - pred)**2))

        print(mae, rmse)

        if mae < best_mae:
            best_mae = mae

        if rmse < best_rmse:
            best_rmse = rmse

print(best_mae, best_rmse)
