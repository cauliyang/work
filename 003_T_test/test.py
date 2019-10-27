# 导入模块
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind as ttest
from scipy.stats import levene as f
from itertools import product


# 定义函数
def init_alp(cutoff):
    alp_lower = [chr(a) for a in range(ord('a'), ord('z') + 1)]
    alp_upper = [chr(a) for a in range(ord('A'), ord('Z') + 1)]
    re_list = []
    if cutoff == 0.05:
        re_list.append(alp_lower[0])
        return re_list, alp_lower
    elif cutoff == 0.01:
        re_list.append(alp_upper[0])
        return re_list, alp_upper


# t_test function
def tt(A, B):
    f_p = f(A, B).pvalue
    if f_p <= 0.05:
        t_p = ttest(A, B, equal_var=False).pvalue
    elif f_p > 0.05:
        t_p = ttest(A, B, equal_var=True).pvalue
    return t_p


# main function
def main_cal(cutoff, ind, re_list, alps, pops, df_t):

    pop_ind = None

    alp_ind = alps.index(re_list[ind])

    fix = pops[ind]
    fix_phe = df_t.set_index("Lines").loc[fix].values.flatten()

    res = pops[ind + 1:]
    for pop in res:

        pop_phe = df_t.set_index("Lines").loc[pop].values.flatten()
        t_p = tt(fix_phe, pop_phe)

        if t_p <= cutoff:

            add_alp = alps[alp_ind + 1]

            re_list.append(add_alp)

            pop_ind = pops.index(pop)

            up_pops = pops[:pop_ind]

            # up trace
            for up_ind, up_pop in enumerate(up_pops):
                up_phe = df_t.set_index("Lines").loc[up_pop].values.flatten()
                t_p_2 = tt(up_phe, pop_phe)
                if t_p_2 <= cutoff:
                    pass
                elif t_p_2 > cutoff:
                    re_list[up_ind] = re_list[up_ind] + add_alp
            break
        elif t_p > cutoff:
            re_list.append(alps[alp_ind])
    return pop_ind, re_list


def main_cal_2(cutoff, df, p):
    re_list, alps = init_alp(cutoff)
    ind = 0
    while True:
        ind, re_list = main_cal(cutoff=cutoff,
                                ind=ind,
                                re_list=re_list,
                                alps=alps,
                                pops=p,
                                df_t=df)
        if ind is None:
            break
    return re_list


def main_cal_3():
    df = pd.read_excel('range.xlsx', sheet_name='Sheet1')

    col_num = df.shape[1]

    df.dropna(thresh=col_num - 1, inplace=True)

    df.fillna(method='ffill', inplace=True)

    traits = df.columns[1:]
    new_file = open('./p_value.txt', 'w')
    new_file.write('L_1\tL_2\tP\tTrait\n')
    # cal p_value

    #   pcik up one trait for experiment
    for trait in traits:
        df_t = df.loc[:, ['Lines', trait]]

        df_re = df_t.groupby(by="Lines").mean().sort_values(
            by=trait, ascending=False).rename({trait: 'Mean'}, axis=1)

        pops = list(df_re.index)

        t_1 = list(product(pops, pops))
        for x, y in t_1:
            x_p = df_t.set_index("Lines").loc[x].values.flatten()
            y_p = df_t.set_index("Lines").loc[y].values.flatten()
            t_p_2 = tt(x_p, y_p)
            new_file.write(f'{x}\t{y}\t{t_p_2}\t{trait}\n')

        re_a = main_cal_2(0.05, df=df_t, p=pops)

        re_b = main_cal_2(0.01, df=df_t, p=pops)

        df_re['p_0.05'] = re_a

        df_re['p_0.01'] = re_b

        df_re['trait'] = trait
        yield df_re
    new_file.close()


def main():
    a = main_cal_3()

    dd = pd.concat(a)

    dd.to_csv('test.csv')