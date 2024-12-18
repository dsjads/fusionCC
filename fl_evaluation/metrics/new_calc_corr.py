from fl_evaluation.metrics.metrics_strategy import CommonStrategy


def new_calc_corr(data, method: CommonStrategy):
    features_list = list(data.columns)[:-1]
    label = list(data.columns)[-1]
    corr_dict = {}

    for feature in features_list:
        corr_dict[feature] = method.calculate(data[feature], data[label])
        # corr_dict[feature] = AMPLE2(data[feature], data[label])
    return corr_dict
