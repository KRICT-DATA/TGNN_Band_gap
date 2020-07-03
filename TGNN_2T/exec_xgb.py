import numpy
import pandas
import xgboost as xgb


dataset_name = 'mp_46744_bandgap'
dim_emb = 128
min_test_error = 1e+8
opt_d = -1
opt_n = -1


data = numpy.array(pandas.read_csv('../data/crystal/c2db/id_target_hse.csv'))
numpy.random.shuffle(data)
num_train_data = int(data.shape[0] * 0.8)
train_data = numpy.array(data[num_train_data:, 1:3], dtype=float)
test_data = numpy.array(data[:num_train_data, 1:3], dtype=float)


for d in range(3, 10):
    for n in [100, 150, 200, 300, 400]:
        model_xgb = xgb.XGBRegressor(max_depth=d, n_estimators=n, subsample=0.8)
        model_xgb.fit(train_data[:, 0].reshape(-1, 1), train_data[:, 1].reshape(-1, 1), eval_metric='mae')
        pred_test = model_xgb.predict(test_data[:, 0].reshape(-1, 1))
        test_error = numpy.mean(numpy.abs(test_data[:, 1] - pred_test))
        test_rmse = numpy.sqrt(numpy.mean((test_data[:, 1] - pred_test)**2))
        print('d={}\tn={}\tMAE: {:.4f}'.format(d, n, test_error))

        if test_error < min_test_error:
            min_test_error = test_error
            min_rmse = test_rmse
            opt_d = d
            opt_n = n

    print(min_test_error, min_rmse)

# model_xgb = xgb.XGBRegressor(max_depth=opt_d, n_estimators=opt_n, subsample=0.8)
# model_xgb.fit(train_data_x, train_data_y, eval_metric='mae', eval_set=[(train_data_x, train_data_y)])
# pred_test = model_xgb.predict(test_data_x)
# print('opt d={}\topt n={}\tmin MAE: {:.4f}'.format(opt_d, opt_n, min_test_error))
#
# reg_result = numpy.hstack([test_data_id.reshape(-1, 1), test_data_y.reshape(-1, 1), pred_test.reshape(-1, 1)])
# numpy.savetxt('pred_result/pred_' + dataset_name + '_dml_xgb.csv', reg_result, delimiter=',')
