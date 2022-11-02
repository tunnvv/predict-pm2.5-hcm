from .utils import *
from .model import *

from . import config as c

def main():
    # get global path to these folders `predict-pm2.5-hcm`
    parent_dir = c.parent_dir
    create_folder_for_output(parent_dir)

    # read training data to dataframe
    data_dir = os.path.join(parent_dir, 'data', 'training_data', c.fname_data)
    df = load_data(data_dir=data_dir)
    print(df.columns)

    """ ================ PREPARE DATA TRAINING ================ """
    X, y = generate_time_series_data(df, c.window_size, c.stride_pred)
    X = normalize_data(X)
    print("Checking correctness of sample generation ::")
    print(f"The shape of a sample is {X[0].shape}, the correct shape is ({c.window_size}, {c.num_feature}))")

    # split data in to train/validation
    x_train, y_train, x_test, y_test = split_data(X, y, c.train_ratio)

    """ ================== LSTM ====================== """
    lstm = LTSMModel(params=c.lstm_params)
    lstm.fit_and_save(X=x_train, y=y_train)

    # predict testset using the trained LSTM model
    y_pred1 = lstm.predict(x_test).flatten()

    # store predict x_train, using for compute weight sharing of two model
    ytr_pred1 = lstm.predict(x_train).flatten()

    """ ================= LGBM ======================== """
    # prepare data for lgbm: flatten data point
    _, fn = x_train[0].shape
    lgbm = LightGBMModel(c.lgbm_params)
    x_train, x_test = x_train.reshape(-1, ws * fn), x_test.reshape(-1, ws * fn)

    lgbm.fit(X=x_train, y=y_train)
    lgbm.save_model(c.lgbm_output)

    # predict testset using the trained LGBM model
    y_pred2 = lgbm.predict(x_test).flatten()

    # store predict x_train, using for compute weight sharing of two model
    ytr_pred2 = lgbm.predict(x_train).flatten()

    """ ============== LSTM-TSLightGBM ================== """
    w1, w2 = compute_weight_sharing(ytr_pred1, ytr_pred2, y_train)
    y_pred = w1 * y_pred1 + w2 * y_pred2

    """ =================== METRIC ====================== """
    df_metric = create_metrics_report_table(['LSTM', 'LightGBM', 'LSTM-TSLightGBM'],
                                            [y_pred1, y_pred2, y_pred], [y_test] * 3)
    df_metric.to_csv(os.path.join(parent_dir, 'output', 'results', 'metrics.csv'), index=False)


if __name__ == '__main__':
    main()
