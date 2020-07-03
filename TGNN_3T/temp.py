import numpy
import pandas
import ase.db
import util.mp_downloader as mp

# con = ase.db.connect('mp_gllbsc.db')
# print(con.columnnames)
#
# data = list()
# for row in con.select():
#     data.append(['mp-' + str(int(row.mpid)), row.gllbsc_ind_gap])
#     print(row)
#
# print(len(data))
# numpy.savetxt('id_target.csv', numpy.vstack(data), delimiter=',', fmt='%s,%s')


# data = numpy.array(pandas.read_csv('id_target_raw.csv'))
# vec_data = list()
#
# for d in data:
#     c = mp.download_mp_data(d[1], d[2], d[0], d[3], d[4], d[5], '../data/crystal/prb')
#
#     if c is not None:
#         vec_data.append(c)
#
#     df = pandas.DataFrame.from_records(vec_data, columns=None)
#     df.to_csv('../data/crystal/prb/id_target.csv', index=None, header=None)


emb = numpy.array(pandas.read_csv('emb.csv', header=None))
data = numpy.array(pandas.read_csv('id_target.csv'))
dim_emb = 128
meta_data = list()

for i in range(0, emb.shape[0]):
    target = emb[i, 128]
    idx = int(emb[i, 129])

    meta_data.append([data[idx, 0], idx, data[idx, 3], data[idx, 1], data[idx, 4], data[idx, 5], data[idx, 7]])

meta_data = numpy.array(meta_data)
numpy.savetxt('metadata.csv', meta_data, delimiter=',', fmt='%s,%s,%s,%s,%s,%s,%s')
