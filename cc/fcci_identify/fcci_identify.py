import numpy as np
from skfuzzy import control as ctrl, gaussmf


def build_expert_model():
    SS = ctrl.Antecedent(np.arange(0, 1, 0.01), 'SS')
    SF = ctrl.Antecedent(np.arange(0, 1, 0.01), 'SF')
    CR = ctrl.Antecedent(np.arange(0, 1, 0.01), 'CR')
    CC = ctrl.Consequent(np.arange(0, 1, 0.01), 'CC')

    centers = [0,0.25,0.5,0.75,1]
    sigma = 0.01
    for i, center in enumerate(centers):
        label = ['poor', 'mediocre', 'average', 'decent', 'good'][i]
        SS[label] = gaussmf(SS.universe, center, sigma)
        SF[label] = gaussmf(SF.universe, center, sigma)
        CR[label] = gaussmf(CR.universe, center, sigma)
        CC[label] = gaussmf(CC.universe, center, sigma)

    rules = [
        ctrl.Rule(SS['poor'], CC['poor']),
        ctrl.Rule(SS['mediocre'], CC['mediocre']),
        ctrl.Rule(SS['average'], CC['average']),
        ctrl.Rule(SS['decent'], CC['decent']),
        ctrl.Rule(SS['good'], CC['good']),

        ctrl.Rule(CR['poor'], CC['poor']),
        ctrl.Rule(CR['mediocre'], CC['mediocre']),
        ctrl.Rule(CR['average'], CC['average']),
        ctrl.Rule(CR['decent'], CC['decent']),
        ctrl.Rule(CR['good'], CC['good']),

        ctrl.Rule(SF['poor'], CC['poor']),
        ctrl.Rule(SF['mediocre'], CC['mediocre']),
        ctrl.Rule(SF['average'], CC['average']),
        ctrl.Rule(SF['decent'], CC['decent']),
        ctrl.Rule(SF['good'], CC['good'])
    ]

    activity_ctrl = ctrl.ControlSystem(rules)
    activity_simulation = ctrl.ControlSystemSimulation(activity_ctrl)
    return activity_simulation


if __name__ == '__main__':
    model = build_expert_model()
    # 为输入变量设置值
    model.input['SS'] = 0.7
    model.input['SF'] = 0.6
    model.input['CR'] = 0.8
    # 运行推理
    model.compute()
    # 输出结果
    print(f"推理得到的 CC 值: {model.output['CC']}")
