import numpy as np

from fl_evaluation.metrics.metrics_strategy import CommonStrategy


def get_N_para(feature, label):
    # non-binary
    feature = np.array(feature)
    label = np.array(label)
    EPSILON = 1e-4

    label_f = label[np.fabs(label) > EPSILON]
    feature_f = feature[np.fabs(label) > EPSILON]
    Ncf = np.sum(feature_f * label_f)
    Nuf = np.sum(label_f[np.fabs(feature_f) <= EPSILON])

    # label_s = label[label == 0]
    feature_s = feature[np.fabs(label) <= EPSILON]
    Ncs = np.sum(feature_s)
    Nus = np.sum(np.array(np.fabs(feature_s) <= EPSILON, dtype=int))

    return Ncf, Nuf, Ncs, Nus


class DStar(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        if Ncs + Nuf ==0 :
            return 0
        else:
            return Ncf ** 2 / (Ncs + Nuf)


class DStarSub1(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return 1 / (Ncs + Nuf)


class Ochiai(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        div = (Ncf + Nuf) * (Ncf + Ncs)
        div = 0 if div < 0 else div
        if div == 0:
            return 0
        else:
            return Ncf / np.sqrt(div)



class OchiaiSubOne(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return 1 / np.sqrt(Ncf + Nuf)


class OchiaiSubTwo(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        if Ncf + Ncs:
            return 0
        else:
            return 1 / np.sqrt(Ncf + Ncs)


class Barinel(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        if Ncs + Ncf==0:
            return 0
        else:
            return 1 - Ncs / (Ncs + Ncf)


class ER1(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf - Ncs / (Ncs + Nus + 1)


class ER2(CommonStrategy):
    def calculate(self, feature, label):
        # ER2
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf / (Nuf + Ncs + Nus)

# ER3
class ER3(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return (Ncf / (Ncf + Nuf)) / ((Ncf / (Ncf + Nuf)) + (Ncs / (Ncs + Nus)))

class ER4(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf - Ncs

# ER5
class ER5(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf / (Ncf + Nuf + Ncs + Nus)

class ER6(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        # Rogot1
        return ((Ncf / (2 * Ncf + Nuf + Ncs)) + (Nus / (2 * Nus + Nuf + Ncs))) / 2

# Kulczynski2
class Kulczynski2(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return ((Ncf / (Ncf + Nuf)) + (Ncf / (Ncf + Ncs))) / 2

# Jaccard
class Jaccard(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        div = Ncf + Nuf + Ncs
        return Ncf / div

# Jaccard_s_1
class Jaccard_sub_one(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        div = Ncf + Nuf + Ncs
        return 1 / div


class Tarantula(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        tf = Ncf + Nuf
        tp = Ncs + Nus
        return (Ncf / tf) / (Ncf / tf + Ncs / tp)

class Tarantula_sub_one(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        tf = Ncf + Nuf
        tp = Ncs + Nus
        return 1 / (Ncf / tf + Ncs / tp)


class Naish1(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        if Nuf > 0:
            return -1
        else:
            return Nus

class Binary(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        if Nuf > 0:
            return 0
        else:
            return 1

class CrossTab(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf

class M2(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf / (Ncf + Nus + 2 * (Nuf + Ncs))

class AMPLE2(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf / (Ncf + Nuf) - Ncs / (Ncs + Nus)


class Wong3(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        if Ncs <= 2:
            h = Ncs
        elif Ncs > 2 and Ncs <= 10:
            h = 2 + 0.1 * (Ncs - 2)
        else:
            h = 2.8 + 0.001 * (Ncs - 10)
        return Ncf - h

# Arithmetic Mean
class AM(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return (2 * Ncf * Ncs - 2 * Nuf * Ncs) / ((Ncf + Ncs) * (Nus + Nuf) + (Ncf + Nuf) * (Ncs + Nus))

class Cohen(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return (2 * Ncf * Ncs - 2 * Nuf * Ncs) / ((Ncf + Ncs) * (Nus + Ncs) + (Ncf + Nuf) * (Nuf + Nus))

class Fleiss(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return (4 * Ncf * Nus - 4 * Nuf * Ncs - (Nuf - Ncs) ** 2) / (2 * Ncf + Nuf + Ncs + 2 * Nus + Nuf + Ncs)

class GP02(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        Nus = 0 if Nus < 0 else Nus
        Ncs = 0 if Ncs < 0 else Ncs
        return 2 * (Ncf + np.sqrt(Nus)) + np.sqrt(Ncs)

class GP03(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        Ncs = 0 if Ncs < 0 else Ncs
        return np.sqrt(np.abs(Ncf * Ncf - np.sqrt(Ncs)))

class GP13(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf + Ncf / (2 * Ncs + Ncf)

class GP13_sub_one(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return 1 / (2 * Ncs + Ncf)

class GP13_sub_two(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf / (2 * Ncs + Ncf)

class GP19(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf * np.sqrt(np.abs(Ncs - Ncf + Nuf - Nus))

class Op2(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncf - Ncs / (Ncs + Nus + 1)


class Op2_sub_one(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return Ncs / (Ncs + Nus + 1)


class Op2_sub_two(CommonStrategy):
    def calculate(self, feature, label):
        Ncf, Nuf, Ncs, Nus = get_N_para(feature, label)
        return 1 / (Ncs + Nus + 1)





